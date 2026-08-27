from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .congestion_model import CongestionModel
from .demand_model import NonStationaryDemandModel
from .metrics import jain_fairness, normalized_imbalance
from .utils import project_root


@dataclass
class Slot:
    slot_id: int
    zone: int
    aisle: int
    x: float
    y: float
    level: int
    capacity: int
    category: str


class WarehouseDigitalTwin:
    """Event-driven digital twin for the shared warehouse described in the manuscript.

    Operational state uses second-level timestamps while allocation decisions occur at the
    configured five-minute interval. This preserves the manuscript's 1 s synchronization
    resolution without forcing an expensive RL action every second.
    """

    def __init__(self, cfg: dict, seed: int, dynamic_twin: bool = True):
        self.cfg = cfg
        w = cfg["warehouse"]
        self.num_zones = int(w["zones"])
        self.num_aisles = int(w["aisles"])
        self.num_locations = int(w["storage_locations"])
        self.num_tenants = int(w["tenants"])
        self.dt = int(w["decision_interval_seconds"])
        self.max_candidates = int(w["max_candidate_locations"])
        self.aisle_capacity = int(w["aisle_capacity"])
        self.dynamic_twin = bool(dynamic_twin)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.layout = self._load_layout()
        self.capacity = self.layout["capacity"].to_numpy(dtype=np.int32)
        self.occupancy = np.zeros(self.num_locations, dtype=np.int32)
        self.owner = np.full(self.num_locations, -1, dtype=np.int16)
        self.slot_category = self.layout["category"].astype(str).to_numpy()
        self.compatibility = self._load_compatibility()

        self.demand = NonStationaryDemandModel(cfg["demand"], self.dt, seed + 101)
        self.congestion = CongestionModel(self.num_aisles, cfg["congestion"], seed + 202)
        self.step_index = 0
        self.elapsed_seconds = 0
        self.last_intensity = np.zeros(self.num_tenants, dtype=np.float32)
        self.last_outbound = np.zeros(self.num_tenants, dtype=np.int32)
        self.blocked_fraction = 0.0
        self.demand_multiplier = 1.0
        self.sensor_noise_std = 0.0
        self.observation_delay_steps = 0
        self._state_history: List[np.ndarray] = []

    def _load_layout(self) -> pd.DataFrame:
        path = project_root() / "data" / "warehouse_layout.csv"
        if not path.exists():
            raise FileNotFoundError("warehouse_layout.csv missing. Run generate_dataset.py first.")
        df = pd.read_csv(path)
        required = {"slot_id", "zone", "aisle", "x", "y", "level", "capacity", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Layout missing columns: {sorted(missing)}")
        if len(df) != self.num_locations:
            raise ValueError(f"Expected {self.num_locations} locations, found {len(df)}")
        return df.sort_values("slot_id").reset_index(drop=True)

    def _load_compatibility(self) -> Dict[int, set[str]]:
        path = project_root() / "data" / "product_compatibility.csv"
        if not path.exists():
            return {t: {"general", "fragile", "bulk", "fast"} for t in range(self.num_tenants)}
        df = pd.read_csv(path)
        out: Dict[int, set[str]] = {}
        for tenant, grp in df.groupby("tenant_id"):
            out[int(tenant)] = set(grp.loc[grp["allowed"].astype(bool), "category"].astype(str))
        return out

    def reset(self, scenario: dict | None = None) -> None:
        scenario = scenario or {}
        self.rng = np.random.default_rng(self.seed)
        self.occupancy.fill(0)
        self.owner.fill(-1)
        self.step_index = 0
        self.elapsed_seconds = 0
        self.demand_multiplier = float(scenario.get("demand_multiplier", 1.0))
        self.blocked_fraction = float(scenario.get("blocked_aisle_fraction", 0.0))
        self.sensor_noise_std = float(scenario.get("sensor_noise_std", 0.0))
        self.observation_delay_steps = int(scenario.get("observation_delay_steps", 0))
        self.congestion.reset(self.blocked_fraction)
        self._initialize_inventory()
        self._state_history = []

    def _initialize_inventory(self) -> None:
        target_util = 0.72
        tenant_share = np.asarray([0.14, 0.19, 0.26, 0.17, 0.16, 0.08], dtype=float)
        total_units = int(self.capacity.sum() * target_util)
        for tenant in range(self.num_tenants):
            need = int(total_units * tenant_share[tenant])
            compatible = np.array([
                i for i, cat in enumerate(self.slot_category)
                if cat in self.compatibility.get(tenant, {cat})
            ], dtype=int)
            self.rng.shuffle(compatible)
            for idx in compatible:
                if need <= 0:
                    break
                if self.owner[idx] not in (-1, tenant):
                    continue
                take = min(int(self.capacity[idx]), need)
                self.owner[idx] = tenant
                self.occupancy[idx] = take
                need -= take

    def zone_utilization(self) -> np.ndarray:
        zones = self.layout["zone"].to_numpy(dtype=int)
        util = np.zeros(self.num_zones, dtype=np.float32)
        for z in range(self.num_zones):
            m = zones == z
            util[z] = float(self.occupancy[m].sum() / max(self.capacity[m].sum(), 1))
        return util

    def tenant_occupancy(self) -> np.ndarray:
        return np.array([self.occupancy[self.owner == t].sum() for t in range(self.num_tenants)], dtype=np.float32)

    def fairness(self) -> float:
        # Compare actual tenant shares to manuscript inventory-share profile.
        occ = self.tenant_occupancy()
        target = np.asarray([0.14, 0.19, 0.26, 0.17, 0.16, 0.08], dtype=np.float32)
        target_units = target * max(float(occ.sum()), 1.0)
        ratio = occ / np.maximum(target_units, 1.0)
        return jain_fairness(ratio)

    def candidate_locations(self, tenant: int) -> Tuple[np.ndarray, np.ndarray]:
        allowed = self.compatibility.get(tenant, set(self.slot_category))
        free = self.capacity - self.occupancy
        eligible = np.where((free > 0) & np.isin(self.slot_category, list(allowed)) & ((self.owner == -1) | (self.owner == tenant)))[0]
        if eligible.size == 0:
            return np.full(self.max_candidates, -1, dtype=int), np.zeros(self.max_candidates + 1, dtype=bool)

        zutil = self.zone_utilization()
        zones = self.layout.loc[eligible, "zone"].to_numpy(dtype=int)
        aisles = self.layout.loc[eligible, "aisle"].to_numpy(dtype=int)
        x = self.layout.loc[eligible, "x"].to_numpy(dtype=float)
        y = self.layout.loc[eligible, "y"].to_numpy(dtype=float)
        congestion = self.congestion.state[aisles]
        distance = np.sqrt(np.square(x) + np.square(y))
        score = 0.45 * (distance / (distance.max() + 1e-6)) + 0.35 * congestion + 0.20 * zutil[zones]
        order = eligible[np.argsort(score)]
        chosen = order[: self.max_candidates]
        padded = np.full(self.max_candidates, -1, dtype=int)
        padded[: len(chosen)] = chosen
        mask = np.zeros(self.max_candidates + 1, dtype=bool)
        mask[: len(chosen)] = True
        mask[-1] = True  # explicit no-op
        return padded, mask

    def global_state(self) -> np.ndarray:
        z = self.zone_utilization()
        demand_norm = self.last_intensity / max(float(np.max(self.cfg["demand"]["base_orders_per_hour"])), 1.0)
        c = self.congestion.state.copy()
        occ = self.tenant_occupancy()
        occ_share = occ / max(float(occ.sum()), 1.0)
        s = np.concatenate([z, demand_norm.astype(np.float32), c, occ_share.astype(np.float32)])
        if self.sensor_noise_std > 0:
            s = s + self.rng.normal(0.0, self.sensor_noise_std, len(s))
        s = s.astype(np.float32)
        self._state_history.append(s.copy())
        if self.observation_delay_steps > 0 and len(self._state_history) > self.observation_delay_steps:
            return self._state_history[-1 - self.observation_delay_steps].copy()
        return s

    def local_observation(self, tenant: int) -> np.ndarray:
        z = self.zone_utilization()
        c = self.congestion.state.reshape(self.num_zones, self.num_aisles // self.num_zones).mean(axis=1)
        occ = self.tenant_occupancy()
        own_share = occ[tenant] / max(float(occ.sum()), 1.0)
        intensity = self.last_intensity[tenant] / max(float(self.cfg["demand"]["base_orders_per_hour"][tenant]), 1.0)
        return np.concatenate([z, c, np.array([own_share, intensity, self.fairness()], dtype=np.float32)]).astype(np.float32)

    def _allocate(self, tenant: int, slot_idx: int, units: int) -> tuple[int, float, int]:
        if units <= 0 or slot_idx < 0:
            return 0, 0.0, 0
        free = int(self.capacity[slot_idx] - self.occupancy[slot_idx])
        if free <= 0 or self.owner[slot_idx] not in (-1, tenant):
            return 0, 0.0, 1
        if self.slot_category[slot_idx] not in self.compatibility.get(tenant, {self.slot_category[slot_idx]}):
            return 0, 0.0, 1
        placed = min(free, int(units))
        self.owner[slot_idx] = tenant
        self.occupancy[slot_idx] += placed
        aisle = int(self.layout.at[slot_idx, "aisle"])
        distance = float(np.hypot(self.layout.at[slot_idx, "x"], self.layout.at[slot_idx, "y"]))
        return placed, distance * self.congestion.travel_multiplier(aisle), 0

    def _fulfill_orders(self, tenant: int, orders: int) -> tuple[int, float, float, np.ndarray]:
        if orders <= 0:
            return 0, 0.0, 0.0, np.zeros(self.num_aisles, dtype=float)
        slots = np.where((self.owner == tenant) & (self.occupancy > 0))[0]
        if slots.size == 0:
            return 0, float(orders * 60.0), 0.0, np.zeros(self.num_aisles, dtype=float)
        distances = np.hypot(self.layout.loc[slots, "x"], self.layout.loc[slots, "y"]).to_numpy()
        aisles = self.layout.loc[slots, "aisle"].to_numpy(dtype=int)
        costs = distances * np.array([self.congestion.travel_multiplier(a) for a in aisles])
        sorted_slots = slots[np.argsort(costs)]
        load = np.zeros(self.num_aisles, dtype=float)
        fulfilled = 0
        travel = 0.0
        retrieval = 0.0
        units_per_order = 4
        for _ in range(int(orders)):
            need = units_per_order
            order_travel = 0.0
            for idx in sorted_slots:
                if need <= 0:
                    break
                avail = int(self.occupancy[idx])
                if avail <= 0:
                    continue
                take = min(avail, need)
                self.occupancy[idx] -= take
                if self.occupancy[idx] == 0:
                    self.owner[idx] = -1
                need -= take
                a = int(self.layout.at[idx, "aisle"])
                d = float(np.hypot(self.layout.at[idx, "x"], self.layout.at[idx, "y"]))
                mult = self.congestion.travel_multiplier(a)
                order_travel += d * mult
                load[a] += 1.0
            if need == 0:
                fulfilled += 1
                travel += order_travel
                retrieval += 1.5 + 0.22 * order_travel
            else:
                retrieval += 60.0
        return fulfilled, retrieval, travel, load

    def step(self, selected_slots: List[int]) -> Dict[str, float | np.ndarray]:
        snapshot = self.demand.sample(self.step_index, self.demand_multiplier)
        self.last_intensity = snapshot.demand_intensity.astype(np.float32)
        self.last_outbound = snapshot.outbound_orders.copy()

        inbound_distance = 0.0
        invalid = 0
        for tenant, slot in enumerate(selected_slots):
            _, dist, bad = self._allocate(tenant, int(slot), int(snapshot.inbound_units[tenant]))
            inbound_distance += dist
            invalid += bad

        total_fulfilled = 0
        total_retrieval = 0.0
        total_travel = inbound_distance
        aisle_loads = np.zeros(self.num_aisles, dtype=float)
        for tenant in range(self.num_tenants):
            f, r, d, loads = self._fulfill_orders(tenant, int(snapshot.outbound_orders[tenant]))
            total_fulfilled += f
            total_retrieval += r
            total_travel += d
            aisle_loads += loads

        if self.dynamic_twin:
            congestion = self.congestion.update(aisle_loads, self.aisle_capacity)
        else:
            congestion = self.congestion.state.copy()

        requested = max(int(snapshot.outbound_orders.sum()), 1)
        mean_retrieval = total_retrieval / requested
        throughput = total_fulfilled / (self.dt / 3600.0)
        fair = self.fairness()
        imbalance = normalized_imbalance(self.tenant_occupancy())

        self.step_index += 1
        self.elapsed_seconds += self.dt
        return {
            "retrieval_time": float(mean_retrieval),
            "travel_distance": float(total_travel / requested),
            "throughput": float(throughput),
            "fairness": float(fair),
            "imbalance": float(imbalance),
            "invalid_actions": float(invalid),
            "congestion": congestion.copy(),
            "demand_intensity": snapshot.demand_intensity.copy(),
            "requested_orders": float(snapshot.outbound_orders.sum()),
            "fulfilled_orders": float(total_fulfilled),
        }
