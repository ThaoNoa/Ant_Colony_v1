"""
trainer.py — Headless Training Pipeline
Python translation of trainer.js

Chạy DDPG cực nhanh mà không cần render DOM/Canvas.
"""

import math
import asyncio
import time
from typing import Callable, Awaitable

from graph import Graph
from aco_engine import ACOEngine
from environment import TSPEnvironment


class HeadlessTrainer:
    def __init__(self, graph: Graph, agent):
        self.graph = graph
        self.agent = agent
        # Dùng ACOEngine và Environment riêng để không can thiệp UI state
        self.aco = ACOEngine(graph, num_ants=30)
        self.env = TSPEnvironment(graph, self.aco)

    async def train(
        self,
        num_gens: int = 2000,
        on_progress: Callable | None = None,
    ):
        """Chạy huấn luyện tốc độ cao.

        Args:
            num_gens:    Số generation cần chạy.
            on_progress: async callback(gen, best_cost, actor_loss, noise)
        """
        self.env.reset()
        self.agent.reset_noise()
        self.graph.reset_pheromones()

        # Lưu tắc đường hiện tại để phục hồi sau khi train
        saved_blocked = set(self.graph.blocked_edges)
        self.graph.blocked_edges.clear()
        self.graph._compute_distances()

        gen_since_last_train = 0
        start_time = time.time()

        for gen in range(num_gens):
            state      = self.env.get_state()
            action_info = self.agent.select_action(state)
            alpha, rho = action_info['alpha'], action_info['rho']
            raw_action  = action_info['raw_action']

            self.aco.set_params(alpha, rho)
            if self.env.stuck_counter >= 12:
                self.graph.reset_pheromones()

            result     = self.aco.run_iteration()
            round_cost = result['best_cost']
            next_result = self.env.step(round_cost, alpha, rho)
            next_state  = next_result['next_state']

            self.agent.remember(state, raw_action, next_result['reward'], next_state)

            gen_since_last_train += 1
            if gen_since_last_train >= 2:
                gen_since_last_train = 0
                self.agent.train_step()

            # Progress callback mỗi 50 gen + nhả coroutine để không block
            if gen > 0 and gen % 50 == 0:
                if on_progress:
                    best = self.env.best_cost if math.isfinite(self.env.best_cost) else None
                    await on_progress(gen, best,
                                      self.agent.last_actor_loss,
                                      self.agent.noise_level)
                await asyncio.sleep(0)  # yield to event loop

        elapsed = time.time() - start_time
        print(f'[Trainer] Đã train {num_gens} gens trong {elapsed:.2f}s.')

        # Phục hồi tắc đường
        for key in saved_blocked:
            self.graph.blocked_edges.add(key)
        self.graph._compute_distances()
        self.graph.reset_pheromones()

        if on_progress:
            best = self.env.best_cost if math.isfinite(self.env.best_cost) else None
            await on_progress(num_gens, best,
                              self.agent.last_actor_loss,
                              self.agent.noise_level)
