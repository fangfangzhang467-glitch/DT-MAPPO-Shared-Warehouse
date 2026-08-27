from __future__ import annotations

from typing import Iterable
import torch
from torch import nn


def activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    return nn.ReLU


def mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int, act: str = "relu") -> nn.Sequential:
    layers = []
    dims = [input_dim, *list(hidden_sizes)]
    A = activation(act)
    for a, b in zip(dims[:-1], dims[1:]):
        linear = nn.Linear(a, b)
        nn.init.orthogonal_(linear.weight, gain=2 ** 0.5)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, A()])
    out = nn.Linear(dims[-1], output_dim)
    nn.init.orthogonal_(out.weight, gain=0.01)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class MaskedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes, act: str = "relu"):
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, action_dim, act)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(obs)
        if action_mask is not None:
            mask = action_mask.bool()
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes, act: str = "relu"):
        super().__init__()
        self.net = mlp(state_dim, hidden_sizes, 1, act)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixer(nn.Module):
    """Monotonic QMIX mixer with hypernetworks conditioned on the global state."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [B, N], states: [B, S]
        b = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(b, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(b, 1, self.embed_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_qs.view(b, 1, self.num_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(b, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(b, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(b)
