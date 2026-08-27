from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .networks import CentralizedCritic, MaskedActor


@dataclass
class RolloutBatch:
    states: np.ndarray
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_value: float


class MAPPO:
    """On-policy Multi-Agent PPO with centralized critic and decentralized actors."""

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, num_agents: int, cfg: dict, device: str = "cpu"):
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.cfg = cfg
        hidden = cfg["hidden_sizes"]
        act = cfg.get("activation", "relu")
        self.actors = nn.ModuleList([MaskedActor(obs_dim, action_dim, hidden, act) for _ in range(num_agents)]).to(self.device)
        self.critic = CentralizedCritic(state_dim, hidden, act).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actors.parameters(), lr=float(cfg["learning_rate"]))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["learning_rate"]))
        self.gamma = float(cfg["gamma"])
        self.lam = float(cfg["gae_lambda"])
        self.clip = float(cfg["clip_ratio"])
        self.entropy_coef = float(cfg["entropy_coef"])
        self.value_coef = float(cfg["value_coef"])
        self.max_grad_norm = float(cfg["max_grad_norm"])

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        actions, logps = [], []
        for i, actor in enumerate(self.actors):
            logits = actor(obs_t[i:i+1], masks_t[i:i+1])
            dist = Categorical(logits=logits)
            a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            actions.append(int(a.item()))
            logps.append(float(dist.log_prob(a).item()))
        return np.asarray(actions, dtype=np.int64), np.asarray(logps, dtype=np.float32)

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(s).item())

    def compute_gae(self, batch: RolloutBatch):
        t_steps = len(batch.rewards)
        adv = np.zeros(t_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(t_steps)):
            next_v = batch.next_value if t == t_steps - 1 else batch.values[t + 1]
            nonterminal = 1.0 - float(batch.dones[t])
            delta = batch.rewards[t] + self.gamma * next_v * nonterminal - batch.values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + batch.values
        if self.cfg.get("normalize_advantage", True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(self, batch: RolloutBatch) -> dict:
        adv, returns = self.compute_gae(batch)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(batch.masks, dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        n = states.shape[0]
        mb = min(int(self.cfg["minibatch_size"]), n)
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(int(self.cfg["update_epochs"])):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                ratios, entropy_terms = [], []
                for agent, actor in enumerate(self.actors):
                    logits = actor(obs[idx, agent], masks[idx, agent])
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(actions[idx, agent])
                    ratio = torch.exp(new_logp - old_logp[idx, agent])
                    ratios.append(ratio)
                    entropy_terms.append(dist.entropy())
                ratio_mean = torch.stack(ratios, dim=1).mean(dim=1)
                entropy = torch.stack(entropy_terms, dim=1).mean()
                surr1 = ratio_mean * adv_t[idx]
                surr2 = torch.clamp(ratio_mean, 1.0 - self.clip, 1.0 + self.clip) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                values = self.critic(states[idx])
                if self.cfg.get("value_clip", True):
                    clipped = old_values[idx] + torch.clamp(values - old_values[idx], -self.clip, self.clip)
                    loss_u = torch.square(values - ret_t[idx])
                    loss_c = torch.square(clipped - ret_t[idx])
                    critic_loss = 0.5 * torch.max(loss_u, loss_c).mean()
                else:
                    critic_loss = 0.5 * torch.square(values - ret_t[idx]).mean()
                self.critic_optim.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path: str) -> None:
        torch.save({"actors": self.actors.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt["actors"])
        self.critic.load_state_dict(ckpt["critic"])


def collect_rollout(env, model: MAPPO, steps: int) -> tuple[RolloutBatch, list[dict], np.ndarray, np.ndarray, np.ndarray]:
    state, obs, masks = env.reset()
    states, obss, maskss, actions, logps, rewards, dones, values = [], [], [], [], [], [], [], []
    infos = []
    for _ in range(steps):
        a, lp = model.act(obs, masks, deterministic=False)
        v = model.value(state)
        next_state, next_obs, next_masks, r, d, info = env.step(a)
        states.append(state); obss.append(obs); maskss.append(masks); actions.append(a); logps.append(lp)
        rewards.append(float(r[0])); dones.append(bool(d[0])); values.append(v); infos.append(info)
        state, obs, masks = next_state, next_obs, next_masks
        if bool(d[0]):
            state, obs, masks = env.reset()
    next_value = model.value(state)
    batch = RolloutBatch(
        states=np.asarray(states, dtype=np.float32), obs=np.asarray(obss, dtype=np.float32),
        masks=np.asarray(maskss, dtype=bool), actions=np.asarray(actions, dtype=np.int64),
        log_probs=np.asarray(logps, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32), values=np.asarray(values, dtype=np.float32), next_value=next_value,
    )
    return batch, infos, state, obs, masks
