"""Network bodies shared by the learners.

Small, deliberately. The observation is 25-dimensional at Maitri and the action
spaces are smaller still, so nothing here needs the capacity a vision or language
model would; a network too large for the problem just overfits the one synthetic
year it is shown. Two hidden layers of 128 units is enough headroom to be wrong
for a reason other than capacity.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic continuous policy: observation -> action in [-1, 1]^d."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, action_dim], output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Action-value function: (observation, action) -> scalar Q."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = mlp([obs_dim + action_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class QNetwork(nn.Module):
    """State-action values for every discrete commitment: observation -> Q(s, .)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, n_actions])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average the target network towards the online one.

    A hard copy of a freshly-updated critic would make the bootstrap target
    chase itself; the slow-moving target is what keeps DDPG-family training
    stable at all.
    """
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.mul_(1.0 - tau).add_(s_param, alpha=tau)


__all__ = ["Actor", "Critic", "QNetwork", "mlp", "soft_update"]
