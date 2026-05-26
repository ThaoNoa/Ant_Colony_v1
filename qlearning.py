"""
qlearning.py — Q-Learning Agent
Python translation of qlearning.js

State Space (3 trạng thái):
  S0: Đường đi đang cải thiện đều đặn
  S1: Bị kẹt ≥ 10 vòng không có kỷ lục mới
  S2: Đường đi đột ngột tệ đi (khi tắc đường xảy ra)

Action Space (3 hành động):
  A0: Duy trì   -> Alpha=1.0, Rho=0.10
  A1: Khám phá  -> Alpha=0.5, Rho=0.80
  A2: Khai thác -> Alpha=2.0, Rho=0.05
"""

import random
import math

# --------------------------------------------------------------------------
# Constants (dict để dễ serialize JSON)
# --------------------------------------------------------------------------
STATES = {
    'IMPROVING': 0,
    'STUCK':     1,
    'DEGRADED':  2,
}

ACTIONS = {
    'MAINTAIN': 0,
    'EXPLORE':  1,
    'EXPLOIT':  2,
}

ACTION_PARAMS = {
    0: {'alpha': 1.0, 'rho': 0.10, 'label': 'Duy trì (A0)'},
    1: {'alpha': 0.5, 'rho': 0.80, 'label': 'Khám phá (A1)'},
    2: {'alpha': 2.0, 'rho': 0.05, 'label': 'Khai thác (A2)'},
}

STATE_LABELS = {
    0: 'BÌNH THƯỜNG',
    1: 'ĐANG KẸT - TÌM ĐƯỜNG MỚI',
    2: 'PHÁT HIỆN TẮC ĐƯỜNG',
}


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.learning_rate    = cfg.get('learning_rate',   0.3)
        self.discount_factor  = cfg.get('discount_factor', 0.9)
        self.epsilon          = cfg.get('epsilon',         0.2)
        self.epsilon_decay    = cfg.get('epsilon_decay',   0.995)
        self.epsilon_min      = cfg.get('epsilon_min',     0.05)

        num_states = 3
        num_actions = 3
        # Q-Table với optimistic initialization (5.0)
        self.q_table: list[list[float]] = [
            [5.0] * num_actions for _ in range(num_states)
        ]

        self.current_state  = STATES['IMPROVING']
        self.current_action = ACTIONS['MAINTAIN']
        self.last_action    = ACTIONS['MAINTAIN']

        self.stuck_counter   = 0
        self.STUCK_THRESHOLD = 10

        self.prev_best_cost  = math.inf
        self.all_time_best   = math.inf

        self.traffic_event_triggered = False

        self.reward_history: list[float] = []
        self.action_history: list[int]   = []
        self.state_history:  list[int]   = []

    # ------------------------------------------------------------------
    # State detection
    # ------------------------------------------------------------------
    def determine_state(self, current_cost: float) -> int:
        if self.traffic_event_triggered:
            self.traffic_event_triggered = False
            return STATES['DEGRADED']

        if current_cost < self.all_time_best - 0.01:
            self.stuck_counter  = 0
            self.all_time_best  = current_cost
            return STATES['IMPROVING']

        self.stuck_counter += 1
        if self.stuck_counter >= self.STUCK_THRESHOLD:
            return STATES['STUCK']

        if (current_cost > self.all_time_best * 1.2
                and self.all_time_best != math.inf):
            return STATES['DEGRADED']

        return STATES['IMPROVING']

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        q_vals = self.q_table[state]
        return q_vals.index(max(q_vals))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_reward(self, new_cost: float, prev_cost: float) -> float:
        if new_cost < self.all_time_best:
            return 10.0
        if new_cost > prev_cost * 1.05:
            return -10.0
        if self.stuck_counter >= self.STUCK_THRESHOLD:
            return -5.0
        return 1.0

    # ------------------------------------------------------------------
    # Q-Table update (Bellman)
    # ------------------------------------------------------------------
    def update_q_table(self, state: int, action: int,
                       reward: float, next_state: int):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, current_cost: float) -> dict:
        prev_cost = self.prev_best_cost

        new_state  = self.determine_state(current_cost)
        reward     = self.compute_reward(current_cost, prev_cost)

        self.update_q_table(self.current_state, self.current_action,
                            reward, new_state)

        new_action = self.select_action(new_state)

        self.reward_history.append(reward)
        self.action_history.append(new_action)
        self.state_history.append(new_state)

        self.current_state  = new_state
        self.current_action = new_action
        self.prev_best_cost = current_cost
        if current_cost < self.all_time_best:
            self.all_time_best = current_cost

        return {
            'action':   new_action,
            'params':   ACTION_PARAMS[new_action],
            'state':    new_state,
            'reward':   reward,
            'epsilon':  self.epsilon,
            'q_values': list(self.q_table[new_state]),
        }

    # ------------------------------------------------------------------
    # Traffic event
    # ------------------------------------------------------------------
    def trigger_traffic_event(self):
        self.traffic_event_triggered = True
        self.stuck_counter = 0

    # ------------------------------------------------------------------
    # Debug display
    # ------------------------------------------------------------------
    def get_q_table_display(self) -> list[dict]:
        state_names  = ['S0:Improving', 'S1:Stuck', 'S2:Degraded']
        action_names = ['A0:Maintain',  'A1:Explore', 'A2:Exploit']
        result = []
        for s, row in enumerate(self.q_table):
            for a, q in enumerate(row):
                result.append({
                    'state':  state_names[s],
                    'action': action_names[a],
                    'q':      round(q, 2),
                })
        return result
