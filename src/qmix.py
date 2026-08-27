from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np
import torch
from torch import nn

from .networks import AgentQNetwork, QMixer


@dataclass
class Transition:
    state: np.ndarray
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    reward: float
    next_state: np.ndarray
    next_obs: np.ndarray
    next_masks: np.ndarray
    done: bool


class QMIX:
    """Compact QMIX baseline with replay memory and target networks."""

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, num_agents: int, cfg: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.cfg = cfg
        hidden = int(cfg["agent_hidden"])
        mix_hidden = int(cfg["mixer_hidden"])
        self.agents = nn.ModuleList([AgentQNetwork(obs_dim, action_dim, hidden) for _ in range(num_agents)]).to(self.device)
        self.target_agents = nn.ModuleList([AgentQNetwork(obs_dim, action_dim, hidden) for _ in range(num_agents)]).to(self.device)
        self.mixer = QMixer(num_agents, state_dim, mix_hidden).to(self.device)
        self.target_mixer = QMixer(num_agents, state_dim, mix_hidden).to(self.device)
        self.target_agents.load_state_dict(self.agents.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(list(self.agents.parameters()) + list(self.mixer.parameters()), lr=float(cfg["learning_rate"]))
        self.replay: Deque[Transition] = deque(maxlen=int(cfg["replay_capacity"]))
        self.gamma = float(cfg["gamma"])
        self.batch_size = int(cfg["batch_size"])
        self.train_steps = 0

    def epsilon(self, step: int) -> float:
        start, end, decay = float(self.cfg["epsilon_start"]), float(self.cfg["epsilon_end"]), max(int(self.cfg["epsilon_decay_steps"]), 1)
        f = min(step / decay, 1.0)
        return start + f * (end - start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, step: int = 0, deterministic: bool = False) -> np.ndarray:
        eps = 0.0 if deterministic else self.epsilon(step)
        actions = []
        for i, net in enumerate(self.agents):
            valid = np.flatnonzero(masks[i])
            if len(valid) == 0:
                actions.append(self.action_dim - 1)
                continue
            if random.random() < eps:
                actions.append(int(np.random.choice(valid)))
            else:
                o = torch.as_tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                q = net(o).squeeze(0)
                mask_t = torch.as_tensor(masks[i], dtype=torch.bool, device=self.device)
                q = q.masked_fill(~mask_t, torch.finfo(q.dtype).min)
                actions.append(int(torch.argmax(q).item()))
        return np.asarray(actions, dtype=np.int64)

    def store(self, transition: Transition) -> None:
        self.replay.append(transition)

    def update_targets(self) -> None:
        self.target_agents.load_state_dict(self.agents.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def train_step(self) -> dict:
        if len(self.replay) < self.batch_size:
            return {"loss": float("nan")}
        batch = random.sample(self.replay, self.batch_size)
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(np.stack([b.obs for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([b.actions for b in batch]), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([b.next_obs for b in batch]), dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([b.next_masks for b in batch]), dtype=torch.bool, device=self.device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        chosen_qs, target_qs = [], []
        for i in range(self.num_agents):
            q = self.agents[i](obs[:, i]).gather(1, actions[:, i:i+1]).squeeze(1)
            chosen_qs.append(q)
            with torch.no_grad():
                tq = self.target_agents[i](next_obs[:, i])
                tq = tq.masked_fill(~next_masks[:, i], torch.finfo(tq.dtype).min)
                target_qs.append(tq.max(dim=1).values)
        chosen = torch.stack(chosen_qs, dim=1)
        target = torch.stack(target_qs, dim=1)
        q_tot = self.mixer(chosen, states)
        with torch.no_grad():
            target_tot = self.target_mixer(target, next_states)
            y = rewards + self.gamma * (1.0 - dones) * target_tot
        loss = torch.mean(torch.square(q_tot - y))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agents.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % int(self.cfg["target_update_steps"]) == 0:
            self.update_targets()
        return {"loss": float(loss.detach().cpu())}

    def save(self, path: str) -> None:
        torch.save({"agents": self.agents.state_dict(), "mixer": self.mixer.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(ckpt["agents"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.update_targets()
