"""
environment.py — TSPEnvironment
Python translation of environment.js

State Vector (10 chiều):
  [0] normalized_best_cost
  [1] improvement_rate
  [2] stuck_counter_norm
  [3] avg_pheromone
  [4] pheromone_std
  [5] blocked_edge_ratio
  [6] current_alpha_norm
  [7] current_rho_norm
  [8] generation_progress
  [9] pheromone_entropy
"""

import math

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DDPG_CONSTANTS = {
    'STUCK_THRESHOLD': 15,
    'ALPHA_MIN':       0.5,
    'ALPHA_MAX':       2.5,
    'RHO_MIN':         0.02,
    'RHO_MAX':         0.50,
    'REF_WARMUP_GENS': 10,
}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class TSPEnvironment:
    def __init__(self, graph, aco):
        self.graph = graph
        self.aco   = aco

        self.generation     = 0
        self.best_cost      = math.inf
        self.prev_best_cost = math.inf
        self.all_time_best  = math.inf
        self.stuck_counter  = 0
        self.reference_cost = None
        self._ref_cost_sum   = 0.0
        self._ref_cost_count = 0

        self.current_alpha  = 1.0
        self.current_rho    = 0.1

        self.traffic_triggered = False
        self.reward_history: list[float] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.generation      = 0
        self.best_cost       = math.inf
        self.prev_best_cost  = math.inf
        self.all_time_best   = math.inf
        self.stuck_counter   = 0
        self.reference_cost  = None
        self._ref_cost_sum   = 0.0
        self._ref_cost_count = 0
        self.current_alpha   = 1.0
        self.current_rho     = 0.1
        self.traffic_triggered = False
        self.reward_history  = []

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, round_cost: float, alpha: float, rho: float) -> dict:
        self.prev_best_cost = self.best_cost
        self.current_alpha  = alpha
        self.current_rho    = rho
        self.generation    += 1

        # Warmup reference cost
        C = DDPG_CONSTANTS
        if self._ref_cost_count < C['REF_WARMUP_GENS'] and math.isfinite(round_cost):
            self._ref_cost_sum   += round_cost
            self._ref_cost_count += 1
            if self._ref_cost_count == C['REF_WARMUP_GENS']:
                self.reference_cost = self._ref_cost_sum / self._ref_cost_count

        if round_cost < self.best_cost:
            self.best_cost = round_cost
        if round_cost < self.all_time_best:
            self.all_time_best = round_cost

        if round_cost < self.all_time_best + 0.01:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

        reward = self._compute_reward(round_cost)
        self.reward_history.append(reward)
        next_state = self.get_state()
        return {'reward': reward, 'next_state': next_state}

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    def _compute_reward(self, round_cost: float) -> float:
        if not math.isfinite(self.prev_best_cost) or self.prev_best_cost == math.inf:
            return 1.0

        C = DDPG_CONSTANTS
        improvement_rate = (self.prev_best_cost - round_cost) / self.prev_best_cost

        # 1. ACO result reward
        if self.traffic_triggered and round_cost > self.all_time_best * 1.10:
            self.traffic_triggered = False
            return -15.0
        self.traffic_triggered = False

        if improvement_rate > 0.05:
            reward = 20.0
        elif improvement_rate > 0:
            reward = 10.0
        elif self.stuck_counter >= C['STUCK_THRESHOLD']:
            reward = -5.0
        elif round_cost <= self.all_time_best * 1.02:
            reward = 2.0
        else:
            reward = -1.0

        # 2. Parameter health (bell-curve)
        alpha_norm   = (self.current_alpha - C['ALPHA_MIN']) / (C['ALPHA_MAX'] - C['ALPHA_MIN'])
        alpha_health = math.exp(-((alpha_norm - 0.35) / 0.25) ** 2)

        rho_norm   = (self.current_rho - C['RHO_MIN']) / (C['RHO_MAX'] - C['RHO_MIN'])
        rho_health = math.exp(-((rho_norm - 0.20) / 0.25) ** 2)

        param_score = (alpha_health + rho_health - 1.0) * 2.0
        reward += param_score

        # 3. Pheromone entropy
        stats = self.aco.get_pheromone_stats()
        entropy = stats['entropy']
        if entropy < 0.15:
            reward -= 3.0
        elif 0.25 < entropy < 0.75:
            reward += 1.0

        return reward

    # ------------------------------------------------------------------
    # State vector (10D)
    # ------------------------------------------------------------------
    def get_state(self) -> list[float]:
        C = DDPG_CONSTANTS
        g = self.graph
        n = g.size
        total_edges = n * (n - 1) / 2

        # Feature 0: normalized_best_cost
        ref = self.reference_cost if self.reference_cost else (
            self.best_cost if math.isfinite(self.best_cost) else 1.0
        )
        norm_best = min(3.0, self.best_cost / max(ref, 1.0)) if math.isfinite(self.best_cost) else 1.5

        # Feature 1: improvement_rate
        improvement_rate = 0.0
        if math.isfinite(self.prev_best_cost) and self.prev_best_cost > 0:
            improvement_rate = (self.prev_best_cost - self.best_cost) / self.prev_best_cost
            improvement_rate = max(-1.0, min(1.0, improvement_rate))

        # Feature 2: stuck_counter_norm
        stuck_norm = min(1.0, self.stuck_counter / C['STUCK_THRESHOLD'])

        # Pheromone stats
        stats    = self.aco.get_pheromone_stats()
        avg_ph   = min(1.0, stats['avg']      / g.PHEROMONE_MAX)
        std_ph   = min(1.0, stats['variance'] / g.PHEROMONE_MAX)
        entropy  = stats['entropy']

        # Feature 5: blocked_edge_ratio
        blocked_ratio = min(1.0, len(g.blocked_edges) / total_edges) if total_edges > 0 else 0.0

        # Features 6, 7: current params normalized
        alpha_norm = (self.current_alpha - C['ALPHA_MIN']) / (C['ALPHA_MAX'] - C['ALPHA_MIN'])
        rho_norm   = (self.current_rho   - C['RHO_MIN'])   / (C['RHO_MAX']   - C['RHO_MIN'])

        # Feature 8: generation_progress
        gen_progress = (self.generation % 100) / 100.0

        # Feature 9: pheromone_entropy
        norm_entropy = min(1.0, max(0.0, entropy))

        return [
            norm_best,
            improvement_rate,
            stuck_norm,
            avg_ph,
            std_ph,
            blocked_ratio,
            min(1.0, max(0.0, alpha_norm)),
            min(1.0, max(0.0, rho_norm)),
            gen_progress,
            norm_entropy,
        ]

    # ------------------------------------------------------------------
    # Traffic event
    # ------------------------------------------------------------------
    def trigger_traffic_event(self):
        self.traffic_triggered = True
        self.stuck_counter     = 0
        self.best_cost         = math.inf
        self.prev_best_cost    = math.inf
        self.all_time_best     = math.inf

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_best_cost(self) -> float:
        return self.best_cost

    def get_generation(self) -> int:
        return self.generation
