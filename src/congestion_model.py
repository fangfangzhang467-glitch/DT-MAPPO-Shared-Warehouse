from __future__ import annotations

import numpy as np


class CongestionModel:
    """Aisle-level dynamic congestion with persistence, spillover, and stochastic noise."""

    def __init__(self, num_aisles: int, cfg: dict, seed: int):
        self.num_aisles = num_aisles
        self.beta = float(cfg["sensitivity_beta"])
        self.spillover = float(cfg["spillover"])
        self.decay = float(cfg["decay"])
        self.noise_std = float(cfg["noise_std"])
        self.max_c = float(cfg["max_normalized_congestion"])
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(num_aisles, dtype=np.float32)
        self.blocked = np.zeros(num_aisles, dtype=bool)

    def reset(self, blocked_fraction: float = 0.0) -> np.ndarray:
        self.state.fill(0.0)
        self.blocked.fill(False)
        n_blocked = int(round(self.num_aisles * blocked_fraction))
        if n_blocked > 0:
            idx = self.rng.choice(self.num_aisles, n_blocked, replace=False)
            self.blocked[idx] = True
            self.state[idx] = self.max_c
        return self.state.copy()

    def update(self, aisle_loads: np.ndarray, capacity: float) -> np.ndarray:
        loads = np.asarray(aisle_loads, dtype=float) / max(float(capacity), 1.0)
        left = np.roll(loads, 1)
        right = np.roll(loads, -1)
        spill = 0.5 * self.spillover * (left + right)
        innovation = (1.0 - self.decay) * np.clip(loads + spill, 0.0, self.max_c)
        noise = self.rng.normal(0.0, self.noise_std, self.num_aisles)
        self.state = np.clip(self.decay * self.state + innovation + noise, 0.0, self.max_c)
        self.state[self.blocked] = self.max_c
        return self.state.copy()

    def travel_multiplier(self, aisle_ids) -> float:
        ids = np.atleast_1d(aisle_ids).astype(int) % self.num_aisles
        c = float(np.mean(self.state[ids]))
        return 1.0 + self.beta * (c ** 2)
