from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .warehouse_twin import WarehouseDigitalTwin


class SharedWarehouseMAEnv:
    """Six-agent shared-warehouse environment.

    Each tenant is represented by one decentralized agent. At every five-minute decision
    epoch, an agent chooses one dynamically ranked compatible slot from a fixed-size
    candidate list, or an explicit no-op. The large 9,600-slot physical action space is
    therefore exposed through a stable masked categorical action interface.
    """

    def __init__(self, cfg: dict, seed: int, scenario: str = "normal", ablation: str = "full_model"):
        self.cfg = cfg
        self.seed = seed
        self.scenario_name = scenario
        self.ablation_name = ablation
        self.scenario = cfg["experiments"][scenario]
        self.ablation = cfg["ablation"][ablation]
        self.num_agents = int(cfg["warehouse"]["tenants"])
        self.max_candidates = int(cfg["warehouse"]["max_candidate_locations"])
        self.action_dim = self.max_candidates + 1
        self.max_steps = int(cfg["training"]["max_steps_per_episode"])
        self.reward_cfg = cfg["reward"]
        self.twin = WarehouseDigitalTwin(cfg, seed, dynamic_twin=bool(self.ablation["dynamic_twin"]))
        self.steps = 0
        self._candidate_cache: List[np.ndarray] = []
        self._mask_cache: List[np.ndarray] = []

        self.twin.reset(self.scenario)
        self.global_state_dim = len(self.twin.global_state())
        self.local_obs_dim = len(self.twin.local_observation(0))

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.steps = 0
        self.twin.reset(self.scenario)
        state = self.twin.global_state()
        obs = np.stack([self.twin.local_observation(i) for i in range(self.num_agents)])
        masks = self._refresh_candidates()
        return state, obs, masks

    def _refresh_candidates(self) -> np.ndarray:
        self._candidate_cache = []
        self._mask_cache = []
        for tenant in range(self.num_agents):
            cands, mask = self.twin.candidate_locations(tenant)
            self._candidate_cache.append(cands)
            self._mask_cache.append(mask)
        return np.stack(self._mask_cache).astype(bool)

    def _decode_actions(self, actions: np.ndarray) -> List[int]:
        selected: List[int] = []
        for agent, action in enumerate(np.asarray(actions, dtype=int)):
            if action == self.max_candidates:
                selected.append(-1)
                continue
            if action < 0 or action >= self.max_candidates or not self._mask_cache[agent][action]:
                selected.append(-1)
                continue
            selected.append(int(self._candidate_cache[agent][action]))
        return selected

    def _reward(self, info: Dict[str, float | np.ndarray]) -> float:
        retrieval = float(info["retrieval_time"])
        congestion = float(np.mean(info["congestion"]))
        fairness = float(info["fairness"])
        throughput = float(info["throughput"])
        invalid = float(info["invalid_actions"])

        # Bounded terms make the reward numerically stable across workload regimes.
        retrieval_cost = retrieval / (retrieval + 20.0)
        congestion_cost = congestion
        fairness_cost = 1.0 - fairness if bool(self.ablation["fairness_reward"]) else 0.0
        throughput_benefit = throughput / (throughput + 100.0)
        return float(
            -self.reward_cfg["retrieval_weight"] * retrieval_cost
            -self.reward_cfg["congestion_weight"] * congestion_cost
            -self.reward_cfg["fairness_weight"] * fairness_cost
            -self.reward_cfg["invalid_action_penalty"] * invalid / max(self.num_agents, 1)
            +self.reward_cfg["throughput_bonus"] * throughput_benefit
        )

    def step(self, actions: np.ndarray):
        selected_slots = self._decode_actions(actions)
        info = self.twin.step(selected_slots)
        reward = self._reward(info)
        info["reward"] = reward
        self.steps += 1
        done = self.steps >= self.max_steps
        state = self.twin.global_state()
        obs = np.stack([self.twin.local_observation(i) for i in range(self.num_agents)])
        masks = self._refresh_candidates()
        rewards = np.full(self.num_agents, reward, dtype=np.float32)
        dones = np.full(self.num_agents, done, dtype=bool)
        return state, obs, masks, rewards, dones, info


class SingleAgentWarehouseEnv:
    """Centralized controller baseline over the same digital twin and reward function.

    One policy emits one categorical decision per tenant, preserving the same candidate
    interface and identical operational conditions used by the multi-agent experiments.
    """

    def __init__(self, cfg: dict, seed: int, scenario: str = "normal"):
        self.inner = SharedWarehouseMAEnv(cfg, seed, scenario=scenario, ablation="without_marl")
        self.num_agents = self.inner.num_agents
        self.action_dim = self.inner.action_dim
        self.obs_dim = self.inner.global_state_dim

    def reset(self):
        state, _, masks = self.inner.reset()
        return state, masks

    def step(self, actions: np.ndarray):
        state, _, masks, rewards, dones, info = self.inner.step(actions)
        return state, masks, float(rewards[0]), bool(dones[0]), info
