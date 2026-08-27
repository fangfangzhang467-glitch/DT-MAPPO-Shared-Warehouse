from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass
class DemandSnapshot:
    inbound_units: np.ndarray
    outbound_orders: np.ndarray
    demand_intensity: np.ndarray


class NonStationaryDemandModel:
    """Tenant-level non-stationary Poisson process with daily/weekly cycles and bursts.

    The manuscript specifies a non-stationary Poisson process but not its exact numerical
    parameterization. Values are therefore read from config.yaml and explicitly treated as
    reproducibility defaults.
    """

    def __init__(self, cfg: dict, decision_interval_seconds: int, seed: int):
        self.cfg = cfg
        self.dt_hours = decision_interval_seconds / 3600.0
        self.base = np.asarray(cfg["base_orders_per_hour"], dtype=float)
        self.vol = np.asarray(cfg["volatility"], dtype=float)
        self.daily_amp = float(cfg["daily_period_amplitude"])
        self.weekly_amp = float(cfg["weekly_period_amplitude"])
        self.burst_p = float(cfg["burst_probability"])
        self.burst_mult = float(cfg["burst_multiplier"])
        self.rng = np.random.default_rng(seed)
        self.num_tenants = len(self.base)

    def intensity(self, step: int, demand_multiplier: float = 1.0) -> np.ndarray:
        hour = step * self.dt_hours
        day_phase = 2.0 * math.pi * (hour % 24.0) / 24.0
        week_phase = 2.0 * math.pi * (hour % (24.0 * 7.0)) / (24.0 * 7.0)
        daily = 1.0 + self.daily_amp * (0.55 * np.sin(day_phase - 1.0) + 0.45 * np.sin(2 * day_phase + 0.2))
        weekly = 1.0 + self.weekly_amp * np.sin(week_phase - 0.4)
        hetero = 1.0 + self.rng.normal(0.0, self.vol * 0.12, self.num_tenants)
        burst = np.where(self.rng.random(self.num_tenants) < self.burst_p, self.burst_mult, 1.0)
        lam = self.base * daily * weekly * hetero * burst * demand_multiplier
        return np.clip(lam, 0.05, None)

    def sample(self, step: int, demand_multiplier: float = 1.0) -> DemandSnapshot:
        lam_hour = self.intensity(step, demand_multiplier)
        expected_orders = lam_hour * self.dt_hours
        outbound = self.rng.poisson(expected_orders).astype(int)

        # Inbound replenishment is correlated with outbound activity but includes independent noise.
        replenishment_factor = self.rng.uniform(0.85, 1.20, self.num_tenants)
        inbound_batches = self.rng.poisson(np.maximum(expected_orders * replenishment_factor, 0.05))
        avg_units_per_batch = np.array([18, 22, 24, 20, 26, 30])
        inbound_units = inbound_batches * avg_units_per_batch
        return DemandSnapshot(
            inbound_units=inbound_units.astype(int),
            outbound_orders=outbound.astype(int),
            demand_intensity=lam_hour,
        )
