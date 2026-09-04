"""The two-layer agent: DQN for commitment, SDDPG for dispatch, acting as one.

This is the object the deck describes as "Two-layer Safe DRL agent: DQN for
discrete switching + SDDPG for continuous power dispatch." Both learners see the
same observation and are updated from the same environment step, but they choose
independently and are combined only at the point of forming one `DispatchCommand`
-- there is no shared network trunk, no joint loss. Keeping them separate is
deliberate: commitment and dispatch are different kinds of decision (discrete
with anti-cycling memory, versus continuous), and conflating them into one
network would blur exactly the distinction the architecture is built to respect.

Two interfaces are exposed, for two different roles:

  * `act_training(obs, explore, rng)` -- returns the env's native action `dict`,
    for driving `PolarMicrogridEnv` during training.
  * `act(observation, plant)` -- returns a `DispatchCommand`, satisfying the
    `Controller` protocol so this agent can run inside `run_episode`,
    `compare`, and behind `GuardedController` exactly like the rule-based
    baselines.

Safety is not this class's job. It trains and deploys behind the projection
layer (`allotrope.safety.projection`), either directly via `PolarMicrogridEnv`
or via `GuardedController`, and never on its own.
"""

from __future__ import annotations

import numpy as np

from allotrope.agents.dqn import DQNAgent, DQNConfig
from allotrope.agents.sddpg import SDDPGAgent, SDDPGConfig
from allotrope.config import StationConfig
from allotrope.envs.polar_microgrid import observation_vector, observation_width
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid


class HybridAgent:
    """DQN commitment + SDDPG dispatch, combined into one dispatch decision."""

    def __init__(
        self,
        cfg: StationConfig,
        dqn_config: DQNConfig | None = None,
        sddpg_config: SDDPGConfig | None = None,
        melt_ceiling_multiple: float = 4.0,
        name: str = "hybrid_dqn_sddpg",
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.obs_dim = observation_width(cfg)
        self.n_gensets = len(cfg.gensets)
        self.n_storage = len(cfg.storage)
        self.dispatch_dim = self.n_gensets + self.n_storage + 1

        self.dqn = DQNAgent(self.obs_dim, self.n_gensets, dqn_config)
        self.sddpg = SDDPGAgent(self.obs_dim, self.dispatch_dim, sddpg_config)

        # Duplicated here rather than imported from SafetyProjection, so this
        # module has no dependency on the safety package: an agent should not
        # need to know how it is being protected.
        therm = cfg.thermal
        peak_daily_kwh = cfg.occupancy.summer_crew * therm.water_l_per_person_day * therm.snow_melt_kwh_per_l
        self._melt_ceiling_kw = melt_ceiling_multiple * peak_daily_kwh / 24.0

    def reset(self) -> None:
        return None

    # -- training ------------------------------------------------------------

    def act_training(
        self, obs: np.ndarray, explore: bool = True, rng: np.random.Generator | None = None
    ) -> tuple[dict, int]:
        """One action from each learner, packaged as the environment expects.

        The DQN action index is returned alongside the packaged action because
        `DQNAgent.observe` needs it to credit the transition to the right
        commitment pattern -- the environment's `dict` action only carries the
        pattern itself, not which of the `2**n_gensets` indices produced it.
        """
        rng = rng or np.random.default_rng()
        index, commitment = self.dqn.act(obs, explore=explore, rng=rng)
        dispatch = self.sddpg.act(obs, explore=explore, rng=rng)
        action = {"genset_on": np.array(commitment, dtype=np.int8), "dispatch": dispatch}
        return action, index

    def observe(
        self,
        obs: np.ndarray,
        action: dict,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        commitment_index: int,
    ) -> None:
        self.dqn.observe(obs, commitment_index, reward, next_obs, done)
        self.sddpg.observe(obs, action["dispatch"], reward, next_obs, done)

    def update(self, rng: np.random.Generator) -> dict[str, float]:
        stats = {}
        dqn_stats = self.dqn.update(rng)
        sddpg_stats = self.sddpg.update(rng)
        if dqn_stats:
            stats.update(dqn_stats)
        if sddpg_stats:
            stats.update(sddpg_stats)
        return stats

    def end_episode(self) -> None:
        self.dqn.end_episode()
        self.sddpg.end_episode()

    # -- deployment: the Controller protocol ----------------------------------

    def act(self, observation: dict, plant: PolarMicrogrid) -> DispatchCommand:
        """Greedy (no exploration) action, decoded into a physical command.

        This mirrors `PolarMicrogridEnv.decode_action` exactly, so a policy
        trained inside the environment behaves identically once deployed behind
        `GuardedController` instead of the environment's own safety call.
        """
        power_scale_kw = max(self.cfg.total_genset_kw, 1.0)
        obs = observation_vector(observation, self.cfg, power_scale_kw, self._melt_ceiling_kw)

        _, commitment = self.dqn.act(obs, explore=False)
        dispatch = self.sddpg.act(obs, explore=False)

        n_g, n_s = self.n_gensets, self.n_storage
        loading = dispatch[:n_g]
        storage = dispatch[n_g : n_g + n_s]
        melt = float(dispatch[n_g + n_s])

        setpoints = []
        for k, g in enumerate(self.cfg.gensets):
            span = g.rated_kw - g.min_stable_kw
            setpoints.append(g.min_stable_kw + span * (loading[k] + 1.0) / 2.0)

        battery = []
        for k in range(n_s):
            limit = (
                observation["battery_max_discharge_kw"][k]
                if storage[k] >= 0
                else observation["battery_max_charge_kw"][k]
            )
            battery.append(float(storage[k] * limit))

        melt_kw = self._melt_ceiling_kw * (melt + 1.0) / 2.0

        return DispatchCommand(
            genset_on=tuple(bool(v) for v in commitment),
            genset_setpoint_kw=tuple(float(v) for v in setpoints),
            battery_kw=tuple(battery),
            snow_melt_kw=float(melt_kw),
        )

    # -- persistence -----------------------------------------------------------

    def state_dict(self) -> dict:
        return {"dqn": self.dqn.state_dict(), "sddpg": self.sddpg.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.dqn.load_state_dict(state["dqn"])
        self.sddpg.load_state_dict(state["sddpg"])


__all__ = ["HybridAgent"]
