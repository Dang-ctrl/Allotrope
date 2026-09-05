"""Learned controllers: the agents that propose actions for the safety projection to bound.

Two algorithms, because the control problem is genuinely hybrid (see
`allotrope.envs.polar_microgrid`):

  * `BranchingDQN`  -- which sets are committed. Discrete, one binary decision
    per genset, trained as an action-branching dueling Q-network so the
    number of joint commitments never has to be enumerated.
  * `SDDPG`         -- how hard everything committed is worked, and what
    storage and melting do. Continuous, trained as a stochastic actor with
    twin critics (the "S" is the stochastic Gaussian policy; the critics and
    target networks are DDPG/TD3-style).

`HybridAgent` composes the two into one policy that speaks the same
`.act(observation, plant) -> DispatchCommand` interface as the rule-based
baselines in `allotrope.control.baseline`, so it can be dropped into
`allotrope.sim.runner.run_episode` and compared against them directly.

Every agent proposes; nothing here ever bypasses `allotrope.safety.projection`.
"""

from __future__ import annotations

from allotrope.agents.dqn import BranchingDQN, DQNConfig
from allotrope.agents.hybrid import HybridAgent
from allotrope.agents.replay_buffer import ReplayBuffer, Transition
from allotrope.agents.sddpg import SDDPG, SDDPGConfig

__all__ = [
    "BranchingDQN",
    "DQNConfig",
    "SDDPG",
    "SDDPGConfig",
    "HybridAgent",
    "ReplayBuffer",
    "Transition",
]
