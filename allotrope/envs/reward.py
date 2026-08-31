"""The reward: what the agent is actually being asked to optimise.

Reward design is where a reinforcement-learning project quietly succeeds or
fails, so the reasoning is written down rather than left in the weights.

Three principles shape it:

**Physical units, not tuned constants.** Every term is an amount of something
real -- litres, grams, kilowatt-hours, machine starts -- converted to a common
scale by a stated price. That makes each weight arguable on its merits, and it
means the reward can be read back as an operating cost rather than as a number
that only means something relative to itself.

**Safety is not priced.** Life support and freezing carry penalties large enough
to dominate any conceivable fuel saving, but they are not the mechanism that
keeps the station safe -- the projection layer is. A penalty the agent could
learn to trade against would be exactly the wrong design. These terms exist so
that a policy which somehow reaches an unsafe state learns to leave it, not so
that it learns to avoid it in the first place.

**Wear is real cost.** The efficient rule-based baseline reaches a good load
factor at the price of 307 starts a year against the incumbent's 22. Starting a
cold generating set in Antarctica consumes fuel, consumes life, and is exactly
the kind of cost a reward that counts only fuel will happily run up. It is
priced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Prices, in rupees per physical unit, for everything the agent trades off.

    Fuel is the anchor. Everything else is expressed relative to what a litre of
    Jet A-1 costs *delivered to Antarctica* -- which is the number that matters,
    and which is dominated by the ice-class voyage rather than by the fuel.
    """

    fuel_per_l: float = 250.0
    """Jet A-1 landed at a polar station: fuel plus shipping, not pump price."""

    black_carbon_per_g: float = 40.0
    """A shadow price on deposition over ice. Environmental, not a market rate."""

    genset_start_per_event: float = 1500.0
    """Maintenance life consumed by a cold start, amortised."""

    deposit_growth_per_unit: float = 8000.0
    """Fouling accrued over a full deposit range, as deferred maintenance."""

    curtailment_per_kwh: float = 6.0
    """Renewable energy spilled. Small, but it should never be free."""

    unmet_water_per_kwh: float = 60.0
    """Melting owed at the end of a day and not delivered."""

    unserved_per_kwh: float = 5_000.0
    """Any load shed. Deliberately an order above fuel."""

    critical_unserved_per_kwh: float = 500_000.0
    """Life support. Not tradeable against anything, at any efficiency."""

    freeze_violation_per_step: float = 500_000.0
    """Indoor temperature below its hard floor."""

    scale: float = 1.0 / 5_000.0
    """Maps typical hourly cost into a range that keeps value targets well conditioned."""

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class RewardBreakdown:
    """One step's reward, itemised. Logged so a policy can be explained."""

    fuel: float = 0.0
    black_carbon: float = 0.0
    starts: float = 0.0
    deposit: float = 0.0
    curtailment: float = 0.0
    unmet_water: float = 0.0
    unserved: float = 0.0
    critical_unserved: float = 0.0
    freeze: float = 0.0

    @property
    def total_cost(self) -> float:
        return (
            self.fuel
            + self.black_carbon
            + self.starts
            + self.deposit
            + self.curtailment
            + self.unmet_water
            + self.unserved
            + self.critical_unserved
            + self.freeze
        )

    @property
    def safety_cost(self) -> float:
        """The part of the cost that represents harm rather than inefficiency."""
        return self.unserved + self.critical_unserved + self.freeze

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "total_cost": self.total_cost}


class RewardFunction:
    """Turns one step of plant telemetry into a scalar reward and its breakdown."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def __call__(
        self,
        telemetry: dict,
        dt_h: float,
        deposit_delta: float,
        day_rolled_over: bool = False,
        unmet_water_kwh: float = 0.0,
    ) -> tuple[float, RewardBreakdown]:
        w = self.weights
        breakdown = RewardBreakdown(
            fuel=telemetry["fuel_l"] * w.fuel_per_l,
            black_carbon=telemetry["black_carbon_mg"] / 1000.0 * w.black_carbon_per_g,
            starts=telemetry["genset_starts"] * w.genset_start_per_event,
            # Only growth is charged. Burning deposits off is its own reward, and
            # paying an agent for the reduction would let it farm the cycle.
            deposit=max(deposit_delta, 0.0) * w.deposit_growth_per_unit,
            curtailment=telemetry["curtailed_kw"] * dt_h * w.curtailment_per_kwh,
            unserved=telemetry["unserved_kw"] * dt_h * w.unserved_per_kwh,
            critical_unserved=(
                telemetry["critical_unserved_kw"] * dt_h * w.critical_unserved_per_kwh
            ),
            unmet_water=(unmet_water_kwh * w.unmet_water_per_kwh if day_rolled_over else 0.0),
            freeze=(
                w.freeze_violation_per_step
                if telemetry["indoor_temp_c"] < telemetry.get("min_indoor_temp_c", -1e9)
                else 0.0
            ),
        )
        return -breakdown.total_cost * w.scale, breakdown


__all__ = ["RewardFunction", "RewardWeights", "RewardBreakdown"]
