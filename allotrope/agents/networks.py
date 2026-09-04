"""Network building blocks shared by the DQN and SDDPG agents.

Kept small and un-clever on purpose: every network here is a plain MLP. The
project's difficulty lives in the plant, the reward and the safety
projection, not in exotic architectures over a ~20-dimensional observation.
"""

from __future__ import annotations

import torch
from torch import nn

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class DuelingBranchingQNetwork(nn.Module):
    """One Q-value per (branch, action) pair, without enumerating joint actions.

    Each branch is one genset's binary on/off decision. A shared trunk feeds a
    per-branch dueling head (state value + centred advantage), following the
    action-branching architecture for factored discrete action spaces: the
    trunk lets branches share what they learn about the state, and the
    dueling split stabilises learning because most of the value in this
    problem comes from the state (how much load there is), not from any one
    genset's switch.
    """

    def __init__(
        self, obs_dim: int, n_branches: int, actions_per_branch: int = 2, hidden: int = 128
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.actions_per_branch = actions_per_branch
        self.trunk = mlp([obs_dim, hidden, hidden])
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_heads = nn.ModuleList(
            [nn.Linear(hidden, actions_per_branch) for _ in range(n_branches)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q of shape (batch, n_branches, actions_per_branch)."""
        h = self.trunk(obs)
        value = self.value_head(h)  # (batch, 1)
        qs = []
        for head in self.advantage_heads:
            adv = head(h)  # (batch, actions_per_branch)
            qs.append(value + adv - adv.mean(dim=-1, keepdim=True))
        return torch.stack(qs, dim=1)


class StochasticActor(nn.Module):
    """A squashed-Gaussian policy over the continuous dispatch action.

    Outputs a state-dependent mean and log-std (the latter clamped, then
    exponentiated to a std). `.act` samples from that Gaussian and passes the
    *sample* through tanh, so every action coordinate lands in [-1, 1] --
    exactly the range `PolarMicrogridEnv.decode_action` expects, and one the
    safety projection then bounds further regardless.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden])
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def act(
        self, obs: torch.Tensor, deterministic: bool, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        mean, log_std = self(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        noise = torch.randn(mean.shape, generator=generator, device=mean.device)
        return torch.tanh(mean + std * noise)


class TwinCritic(nn.Module):
    """Two independent Q(s, a) estimators; the minimum of the pair is used as
    the training target, which is what keeps DDPG-family critics from
    drifting into optimistic overestimation as training goes on."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1.0 - tau).add_(s, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


__all__ = [
    "mlp",
    "DuelingBranchingQNetwork",
    "StochasticActor",
    "TwinCritic",
    "soft_update",
    "hard_update",
]
