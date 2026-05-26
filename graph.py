"""
graph.py — Quản lý đồ thị: nodes, edges, pheromone matrix
Python translation of graph.js
"""

import math
import random


class Graph:
    PHEROMONE_INIT = 0.1
    PHEROMONE_MIN  = 0.001
    PHEROMONE_MAX  = 10.0
    TRAFFIC_MULTIPLIER = 100

    def __init__(self, num_nodes: int, canvas_width: int = 800,
                 canvas_height: int = 580, padding: int = 60):
        self.num_nodes    = num_nodes
        self.canvas_width  = canvas_width
        self.canvas_height = canvas_height
        self.padding       = padding

        self.nodes: list[dict]        = []
        self.distances: list[list[float]] = []
        self.costs:     list[list[float]] = []
        self.pheromones: list[list[float]] = []
        self.blocked_edges: set[str]       = set()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self):
        """Sinh ngẫu nhiên các node trên canvas."""
        self.nodes = []
        margin = self.padding
        w = self.canvas_width  - margin * 2
        h = self.canvas_height - margin * 2

        # Depot ở trung tâm
        self.nodes.append({
            'id': 0,
            'x': self.canvas_width  / 2,
            'y': self.canvas_height / 2,
            'label': '🏭',
            'is_depot': True,
        })

        attempts = 0
        while len(self.nodes) < self.num_nodes and attempts < 10_000:
            attempts += 1
            x = margin + random.random() * w
            y = margin + random.random() * h

            too_close = any(
                math.hypot(x - n['x'], y - n['y']) < 55
                for n in self.nodes
            )
            if not too_close:
                self.nodes.append({
                    'id': len(self.nodes),
                    'x': x,
                    'y': y,
                    'label': str(len(self.nodes)),
                    'is_depot': False,
                })

        self._compute_distances()
        self._init_pheromones()
        self.blocked_edges.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_distances(self):
        """Tính ma trận khoảng cách Euclidean."""
        n = len(self.nodes)
        self.distances = [[0.0] * n for _ in range(n)]
        self.costs     = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = math.hypot(
                        self.nodes[i]['x'] - self.nodes[j]['x'],
                        self.nodes[i]['y'] - self.nodes[j]['y'],
                    )
                    self.distances[i][j] = d
                    self.costs[i][j]     = d

    def _init_pheromones(self):
        """Khởi tạo ma trận pheromone đồng đều."""
        n = len(self.nodes)
        self.pheromones = [[self.PHEROMONE_INIT] * n for _ in range(n)]

    # ------------------------------------------------------------------
    # Traffic control
    # ------------------------------------------------------------------
    def block_edge(self, i: int, j: int) -> bool:
        """Áp dụng tắc đường lên cạnh (i, j). Trả về False nếu đã tắc."""
        key = f"{min(i, j)}-{max(i, j)}"
        if key in self.blocked_edges:
            return False
        self.blocked_edges.add(key)
        self.costs[i][j] = self.distances[i][j] * self.TRAFFIC_MULTIPLIER
        self.costs[j][i] = self.distances[j][i] * self.TRAFFIC_MULTIPLIER
        return True

    def unblock_edge(self, i: int, j: int):
        key = f"{min(i, j)}-{max(i, j)}"
        self.blocked_edges.discard(key)
        self.costs[i][j] = self.distances[i][j]
        self.costs[j][i] = self.distances[j][i]

    def clear_all_blocked_edges(self):
        """Xoá toàn bộ tắc đường."""
        self.blocked_edges.clear()
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                self.costs[i][j] = self.distances[i][j]

    def is_edge_blocked(self, i: int, j: int) -> bool:
        return f"{min(i, j)}-{max(i, j)}" in self.blocked_edges

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------
    def update_pheromones(self, rho: float, deposits: list[dict]):
        """Cập nhật pheromone sau mỗi iteration.

        Args:
            rho: Tốc độ bay hơi.
            deposits: List[{'path': list[int], 'quality': float}]
        """
        n = len(self.nodes)
        # Bay hơi
        for i in range(n):
            for j in range(n):
                self.pheromones[i][j] = max(
                    self.PHEROMONE_MIN,
                    self.pheromones[i][j] * (1.0 - rho),
                )
        # Deposit
        for dep in deposits:
            path    = dep['path']
            quality = dep['quality']
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                self.pheromones[a][b] = min(
                    self.PHEROMONE_MAX,
                    self.pheromones[a][b] + quality,
                )
                self.pheromones[b][a] = min(
                    self.PHEROMONE_MAX,
                    self.pheromones[b][a] + quality,
                )

    def reset_pheromones(self):
        """Reset pheromone về mức khởi tạo."""
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                self.pheromones[i][j] = self.PHEROMONE_INIT

    # ------------------------------------------------------------------
    # Path metrics
    # ------------------------------------------------------------------
    def path_cost(self, path: list[int]) -> float:
        """Tổng chi phí (có tắc đường) của lộ trình."""
        return sum(self.costs[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def path_distance(self, path: list[int]) -> float:
        """Tổng khoảng cách thực (không tính tắc đường)."""
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))

    # ------------------------------------------------------------------
    # Pheromone stats (for TSPEnvironment.getState)
    # ------------------------------------------------------------------
    def get_pheromone_stats(self) -> dict:
        """Tính thống kê pheromone toàn đồ thị."""
        n = len(self.nodes)
        vals = [
            self.pheromones[i][j]
            for i in range(n)
            for j in range(n)
            if i != j
        ]
        if not vals:
            return {'avg': 0.0, 'variance': 0.0, 'entropy': 0.0}

        avg      = sum(vals) / len(vals)
        variance = math.sqrt(sum((v - avg) ** 2 for v in vals) / len(vals))

        total = sum(vals)
        if total > 0:
            entropy = -sum(
                (v / total) * math.log(v / total + 1e-10) for v in vals
            ) / math.log(len(vals) + 1e-10)
        else:
            entropy = 0.0

        return {
            'avg':      avg,
            'variance': variance,
            'entropy':  min(1.0, max(0.0, entropy)),
        }

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.nodes)
