"""
evaluator.py — Đánh giá tự động hiệu năng Q-Learning và DDPG
Python translation of evaluator.js + test_suite.js

Chạy 5 Test Cases, so sánh với optimal distance (Brute-force).
"""

import math
import asyncio
from itertools import permutations

from graph import Graph
from aco_engine import ACOEngine
from qlearning import QLearningAgent, STATES, ACTION_PARAMS
from environment import TSPEnvironment

# ─────────────────────────────────────────────────────────────────────────────
# Test Cases (translation of TEST_CASES in test_suite.js)
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        'id':      'TC1',
        'name':    'Normal Map (N=8)',
        'nodes':   8,
        'traffic': [],
    },
    {
        'id':      'TC2',
        'name':    'Hard Traffic (N=10) - 1 Edge',
        'nodes':   10,
        'traffic': [{'type': 'hard', 'edge': [1, 3]}],
    },
    {
        'id':      'TC3',
        'name':    'Hard Traffic (N=10) - 2 Edges',
        'nodes':   10,
        'traffic': [{'type': 'hard', 'edge': [2, 5]},
                    {'type': 'hard', 'edge': [4, 7]}],
    },
    {
        'id':      'TC4',
        'name':    'Dynamic Traffic (N=10) - 1 Edge',
        'nodes':   10,
        'traffic': [{'type': 'dynamic', 'gen': 50, 'edge': [1, 4]}],
    },
    {
        'id':      'TC5',
        'name':    'Dynamic Traffic (N=10) - 2 Edges',
        'nodes':   10,
        'traffic': [{'type': 'dynamic', 'gen': 50,  'edge': [3, 6]},
                    {'type': 'dynamic', 'gen': 100, 'edge': [2, 8]}],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Brute-force Solver (Heap's algorithm, translation of BruteForceSolver)
# ─────────────────────────────────────────────────────────────────────────────
class BruteForceSolver:
    @staticmethod
    def solve(graph: Graph) -> dict:
        """Tính đường đi ngắn nhất bằng brute-force.

        Returns:
            {'optimal_cost': float, 'optimal_path': list[int]}
        """
        n    = graph.size
        dist = graph.distances

        vertices = list(range(1, n))
        min_cost = math.inf
        best_path = None

        def evaluate(perm: list[int]):
            nonlocal min_cost, best_path
            cost = dist[0][perm[0]]
            for i in range(len(perm) - 1):
                cost += dist[perm[i]][perm[i + 1]]
                if cost >= min_cost:
                    return
            cost += dist[perm[-1]][0]
            if cost < min_cost:
                min_cost  = cost
                best_path = [0] + list(perm) + [0]

        # Heap's algorithm
        c = [0] * n
        evaluate(vertices)
        i = 0
        while i < n - 1:
            if c[i] < i:
                if i % 2 == 0:
                    vertices[0], vertices[i] = vertices[i], vertices[0]
                else:
                    vertices[c[i]], vertices[i] = vertices[i], vertices[c[i]]
                evaluate(vertices)
                c[i] += 1
                i = 0
            else:
                c[i] = 0
                i   += 1

        return {'optimal_cost': min_cost, 'optimal_path': best_path}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class Evaluator:
    async def run_test_suite(self, ddpg_agent, on_progress=None) -> list[dict]:
        """Chạy bộ test suite cho cả QL và DDPG.

        Args:
            ddpg_agent:   DDPGAgent đã được khởi tạo.
            on_progress:  async callback(message, percent)

        Returns:
            List kết quả mỗi test case.
        """
        results   = []
        MAX_GENS  = 500

        for i, tc in enumerate(TEST_CASES):
            if on_progress:
                await on_progress(
                    f'Đang chạy {tc["id"]}: {tc["name"]}...',
                    (i / len(TEST_CASES)) * 100,
                )

            # 1. Graph cố định cho test case này
            graph = Graph(tc['nodes'], 800, 600)
            graph.generate()

            # 2. Optimal (Brute-force)
            bf     = BruteForceSolver.solve(graph)
            opt    = bf['optimal_cost']

            # 3. Q-Learning
            ql_res = await self._run_agent(tc, 'ql', graph, None, MAX_GENS)

            # 4. DDPG
            ddpg_res = await self._run_agent(tc, 'ddpg', graph, ddpg_agent, MAX_GENS)

            results.append({
                'id':      tc['id'],
                'name':    tc['name'],
                'optimal': opt,
                'ql':      ql_res,
                'ddpg':    ddpg_res,
            })

            # Yield event loop
            await asyncio.sleep(0.05)

        if on_progress:
            await on_progress('Đánh giá hoàn tất!', 100)

        return results

    # ── Internal headless runner ──────────────────────────────────────────────
    async def _run_agent(self, tc: dict, mode: str,
                         base_graph: Graph, ddpg_agent, max_gens: int) -> dict:
        graph = self._clone_graph(base_graph)
        aco   = ACOEngine(graph, num_ants=30)

        env = None
        ql  = None
        if mode == 'ql':
            ql = QLearningAgent()
        else:
            env = TSPEnvironment(graph, aco)
            env.reset()
            ddpg_agent.reset_noise()

        def apply_traffic(gen: int):
            for t in tc['traffic']:
                if ((t['type'] == 'hard' and gen == 0) or
                        (t['type'] == 'dynamic' and gen == t.get('gen', -1))):
                    graph.block_edge(t['edge'][0], t['edge'][1])
                    if mode == 'ql' and ql:
                        ql.trigger_traffic_event()
                    elif env:
                        env.trigger_traffic_event()

        global_best     = math.inf
        stuck_count_total = 0

        for gen in range(max_gens):
            apply_traffic(gen)

            alpha, rho = 1.0, 0.1

            if mode == 'ql':
                if gen == 0:
                    q_res = {'params': ACTION_PARAMS[0], 'action': 0,
                             'state': STATES['IMPROVING']}
                else:
                    q_res = ql.step(global_best)
                alpha = q_res['params']['alpha']
                rho   = q_res['params']['rho']
                aco.set_params(alpha, rho)
                if q_res['action'] == 1 and q_res['state'] == STATES['DEGRADED']:
                    graph.reset_pheromones()
            else:
                state      = env.get_state()
                action_info = ddpg_agent.select_action(state)
                alpha      = action_info['alpha']
                rho        = action_info['rho']
                aco.set_params(alpha, rho)
                if env.stuck_counter >= 12:
                    graph.reset_pheromones()

            result     = aco.run_iteration()
            round_cost = result['best_cost']

            if round_cost < global_best:
                global_best = round_cost

            if mode == 'ddpg':
                env.step(round_cost, alpha, rho)  # Chỉ suy luận, KHÔNG train

            if round_cost > global_best * 1.05:
                stuck_count_total += 1

            # Yield every 50 gens
            if gen % 50 == 0:
                await asyncio.sleep(0)

        return {
            'best_cost':   global_best,
            'error_pct':   0,  # Tính sau khi có optimal
            'stuck_count': stuck_count_total,
        }

    # ── Clone graph ───────────────────────────────────────────────────────────
    def _clone_graph(self, source: Graph) -> Graph:
        g = Graph(source.size, source.canvas_width,
                  source.canvas_height, source.padding)
        g.nodes = [dict(n) for n in source.nodes]
        g._compute_distances()
        g._init_pheromones()
        return g
