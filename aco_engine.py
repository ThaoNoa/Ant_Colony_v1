"""
aco_engine.py — ACO Engine (Ant Colony Optimization)
Python translation of aco.js
"""

import random
import math


class ACOEngine:
    def __init__(self, graph, alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.1, num_ants: int = 30, **kwargs):
        self.graph    = graph
        self.alpha    = alpha   # Trọng số pheromone
        self.beta     = beta    # Trọng số heuristic
        self.rho      = rho     # Tốc độ bay hơi
        self.num_ants = num_ants
        self.Q        = 100.0   # Hằng số deposit

    # ------------------------------------------------------------------
    # Run one full iteration
    # ------------------------------------------------------------------
    def run_iteration(self) -> dict:
        """Chạy một vòng iteration với toàn bộ đàn kiến.

        Returns:
            {'best_path': list[int], 'best_cost': float, 'all_paths': list}
        """
        all_solutions = []

        for _ in range(self.num_ants):
            path = self._build_path()
            if path is not None:
                cost = self.graph.path_cost(path)
                all_solutions.append({'path': path, 'cost': cost})

        if not all_solutions:
            return {'best_path': None, 'best_cost': math.inf, 'all_paths': []}

        all_solutions.sort(key=lambda s: s['cost'])
        best = all_solutions[0]
        best_path = best['path']
        best_cost = best['cost']

        # Elitist strategy: top 30% deposit
        top_k = max(1, int(self.num_ants * 0.3))
        deposits = [
            {'path': s['path'], 'quality': self.Q / s['cost']}
            for s in all_solutions[:top_k]
        ]
        # Bonus cho kiến tốt nhất
        deposits.append({'path': best_path, 'quality': (self.Q * 3) / best_cost})

        self.graph.update_pheromones(self.rho, deposits)

        return {
            'best_path': best_path,
            'best_cost': best_cost,
            'all_paths': all_solutions,
        }

    # ------------------------------------------------------------------
    # Build one ant's path
    # ------------------------------------------------------------------
    def _build_path(self) -> list[int] | None:
        """Một con kiến xây dựng lộ trình hoàn chỉnh."""
        n = self.graph.size
        start = 0
        visited = [False] * n
        path = [start]
        visited[start] = True

        for _ in range(n - 1):
            current = path[-1]
            nxt = self._select_next_node(current, visited)
            if nxt == -1:
                return None
            path.append(nxt)
            visited[nxt] = True

        path.append(start)  # Quay về depot
        return path

    # ------------------------------------------------------------------
    # Stochastic node selection
    # ------------------------------------------------------------------
    def _select_next_node(self, current: int, visited: list[bool]) -> int:
        """Chọn node tiếp theo theo xác suất pheromone × heuristic."""
        n = self.graph.size
        probabilities = [0.0] * n
        total = 0.0

        for j in range(n):
            if visited[j] or j == current:
                continue
            cost = self.graph.costs[current][j]
            if cost <= 0:
                continue
            pheromone = self.graph.pheromones[current][j]
            heuristic = 1.0 / cost
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities[j] = prob
            total += prob

        if total == 0:
            # Fallback: chọn ngẫu nhiên node chưa thăm
            unvisited = [j for j in range(n) if not visited[j] and j != current]
            return random.choice(unvisited) if unvisited else -1

        # Roulette wheel selection
        threshold = random.random() * total
        for j in range(n):
            if probabilities[j] == 0:
                continue
            threshold -= probabilities[j]
            if threshold <= 0:
                return j

        # Dự phòng: node có prob cao nhất
        best_j, best_p = -1, -1.0
        for j in range(n):
            if probabilities[j] > best_p:
                best_p = probabilities[j]
                best_j = j
        return best_j

    # ------------------------------------------------------------------
    # Parameter setter
    # ------------------------------------------------------------------
    def set_params(self, alpha: float, rho: float):
        self.alpha = alpha
        self.rho   = rho

    # ------------------------------------------------------------------
    # Pheromone stats (delegate to graph)
    # ------------------------------------------------------------------
    def get_pheromone_stats(self) -> dict:
        return self.graph.get_pheromone_stats()