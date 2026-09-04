"""HybridAgent: DQN's commitment decision and SDDPG's dispatch decision, as one policy.

Composes the two learned agents into the same `.act(observation, plant) ->
DispatchCommand` interface the rule-based baselines in
`allotrope.control.baseline` implement, so `allotrope.sim.runner.run_episode`
and `allotrope.sim.runner.compare` can score a learned policy against
`LegacyNPlusOne` and `EfficientRuleBased` without a separate code path.

This class only ever *proposes*, exactly like the rule-based baselines do --
neither `LegacyNPlusOne` nor `EfficientRuleBased` self-applies the safety
projection either. The guarantee is centralised in one place,
`allotrope.safety.fallback.GuardedController`, which is what actually wraps
an agent for deployment or evaluation:

    GuardedController(cfg, agent=HybridAgent(cfg, dqn, sddpg))

Giving `HybridAgent` its own parallel safety path, instead of going through
`GuardedController`, would duplicate logic that is already implemented and
tested, and would let this one agent silently diverge from how every other
controller in the project is guarded. `scripts/run_safety_audit.py` and
`allotrope.evaluate` compare a bare `HybridAgent` (the unguarded control
column) against one wrapped in `GuardedController` (what would actually be
deployed) for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allotrope.config import StationConfig
from allotrope.envs.polar_microgrid import PolarMicrogridEnv
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


@dataclass
class HybridAgent:
    """A bare proposer: DQN's commitment choice + SDDPG's dispatch choice.

    Carries no safety logic of its own -- wrap it in
    `allotrope.safety.fallback.GuardedController` before it ever drives a
    plant that matters. `deterministic=True` (greedy/mean action, no
    exploration noise) is what evaluation and deployment want; training uses
    the underlying `dqn.act` / `sddpg.act` directly with
    `deterministic=False`.
    """

    cfg: StationConfig
    dqn: object  # allotrope.agents.dqn.BranchingDQN
    sddpg: object  # allotrope.agents.sddpg.SDDPG
    deterministic: bool = True
    name: str = "hybrid_dqn_sddpg"

    def __post_init__(self) -> None:
        # A throwaway env of the right shape gives this agent the same
        # observation encoding and action decoding the agents were trained
        # against, without duplicating that logic here.
        self._codec_env = PolarMicrogridEnv(self.cfg, periods=2, apply_safety=False)

    def reset(self) -> None:
        return None

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        obs_vec = self._encode(observation, plant)
        genset_on = self.dqn.act(obs_vec, deterministic=self.deterministic)
        dispatch = self.sddpg.act(obs_vec, deterministic=self.deterministic)
        action = {"genset_on": genset_on.astype(np.int8), "dispatch": dispatch}
        self._codec_env.plant = plant
        return self._codec_env.decode_action(action)

    def _encode(self, observation: dict, plant: PolarMicrogrid) -> np.ndarray:
        self._codec_env.plant = plant
        return self._codec_env._observe()


__all__ = ["HybridAgent"]
