from __future__ import annotations

import numpy as np


class StaticClassBasedSlotting:
    """Turnover/proximity-style fixed heuristic baseline.

    Candidate feasibility is still enforced by the environment, but ranking ignores live
    congestion and chooses the physically nearest feasible location.
    """

    def select(self, env, masks: np.ndarray) -> np.ndarray:
        actions = []
        twin = env.twin
        for tenant in range(env.num_agents):
            cands = env._candidate_cache[tenant]
            valid_actions = np.flatnonzero(masks[tenant, :-1])
            if len(valid_actions) == 0:
                actions.append(env.max_candidates)
                continue
            slot_ids = cands[valid_actions]
            x = twin.layout.loc[slot_ids, "x"].to_numpy(dtype=float)
            y = twin.layout.loc[slot_ids, "y"].to_numpy(dtype=float)
            distance = np.hypot(x, y)
            actions.append(int(valid_actions[np.argmin(distance)]))
        return np.asarray(actions, dtype=np.int64)


class RollingHorizonHeuristic:
    """Dynamic reallocation heuristic using current utilization, congestion and distance."""

    def select(self, env, masks: np.ndarray) -> np.ndarray:
        actions = []
        twin = env.twin
        zutil = twin.zone_utilization()
        for tenant in range(env.num_agents):
            cands = env._candidate_cache[tenant]
            valid_actions = np.flatnonzero(masks[tenant, :-1])
            if len(valid_actions) == 0:
                actions.append(env.max_candidates)
                continue
            ids = cands[valid_actions]
            zones = twin.layout.loc[ids, "zone"].to_numpy(dtype=int)
            aisles = twin.layout.loc[ids, "aisle"].to_numpy(dtype=int)
            x = twin.layout.loc[ids, "x"].to_numpy(dtype=float)
            y = twin.layout.loc[ids, "y"].to_numpy(dtype=float)
            d = np.hypot(x, y)
            d = d / (d.max() + 1e-6)
            c = twin.congestion.state[aisles]
            demand = twin.last_intensity[tenant] / max(float(twin.cfg["demand"]["base_orders_per_hour"][tenant]), 1.0)
            score = 0.45 * d + (0.30 + 0.08 * min(demand, 2.0)) * c + 0.25 * zutil[zones]
            actions.append(int(valid_actions[np.argmin(score)]))
        return np.asarray(actions, dtype=np.int64)


def centralized_observation(state: np.ndarray, num_agents: int) -> np.ndarray:
    """Repeat the full state for all action heads of the centralized PPO baseline."""
    return np.repeat(np.asarray(state, dtype=np.float32)[None, :], num_agents, axis=0)
