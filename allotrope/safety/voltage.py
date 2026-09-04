"""Inverter-level Volt-Watt: the safety layer that acts on a voltage the
analytic projection has no model of.

`allotrope.safety.projection.SafetyProjection` bounds capacity, setpoints,
battery power and discretionary load -- everything a power-*balance* model
can see. It cannot see bus voltage, because `allotrope.sim.plant` doesn't
model one. This module is what sits between the projection and actuation
for a station that *does* have a network model
(`StationConfig.network is not None`, `allotrope.sim.network.
DistributionNetwork`): a real, IEEE 1547-2018-shaped Volt-Watt curve that
curtails renewable export when the network solve says the point of
interconnection would otherwise ride above its ride-through band.

Scope, stated plainly: this is Volt-**Watt** only. Volt-VAr -- the other
half of 1547's inverter response curves, using reactive power rather than
curtailed real power to hold voltage -- needs a reactive-power balance this
project's plant does not carry (see `allotrope.sim.network`'s module
docstring). Implementing a Volt-VAr curve with no reactive power behind it
to act on would be exactly the kind of claim without substance this
project's own rules forbid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from allotrope.config import NetworkConfig, StationConfig
from allotrope.sim.network import DistributionNetwork, VoltageSolution
from allotrope.sim.plant import DispatchCommand


@dataclass(frozen=True)
class VoltWattCurve:
    """The IEEE 1547-2018 default Category I/II Volt-Watt curve.

    Below `v1_pu`: full output. Above `v2_pu`: curtailed to `p2_frac` of
    available power (the standard's own default is a floor, not zero --
    an inverter that unplugs entirely at high voltage removes the very
    generation that would otherwise help hold the bus up as load changes).
    Linear in between, which is what the standard itself specifies.
    """

    v1_pu: float = 1.06
    """[public] IEEE 1547-2018 default Category I/II Volt-Watt V1."""
    v2_pu: float = 1.10
    """[public] IEEE 1547-2018 default V2."""
    p2_frac: float = 0.2
    """[public] IEEE 1547-2018 default minimum power fraction at/above V2."""

    def power_limit_frac(self, voltage_pu: float) -> float:
        """Fraction of available power the curve permits at this bus voltage."""
        if voltage_pu <= self.v1_pu:
            return 1.0
        if voltage_pu >= self.v2_pu:
            return self.p2_frac
        span = self.v2_pu - self.v1_pu
        into_band = (voltage_pu - self.v1_pu) / span
        return 1.0 - into_band * (1.0 - self.p2_frac)


@dataclass(frozen=True)
class VoltageReport:
    """What the inverter layer saw and did on one step. Reported like `SafetyReport`."""

    bus_voltage_pu: dict[str, float]
    converged: bool
    curtailed: bool
    renewable_available_kw: float
    renewable_limit_kw: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "bus_voltage_pu": self.bus_voltage_pu,
            "converged": self.converged,
            "curtailed": self.curtailed,
            "renewable_available_kw": self.renewable_available_kw,
            "renewable_limit_kw": self.renewable_limit_kw,
        }


class InverterVoltageLayer:
    """Solves one voltage snapshot per step and applies Volt-Watt to the renewables bus.

    Constructed only for a station whose config declares a `network:``
    section; `allotrope.safety.fallback.GuardedController` takes one as an
    optional extra so every existing caller that doesn't pass one sees
    identical behaviour to before this layer existed.
    """

    def __init__(
        self,
        network_cfg: NetworkConfig,
        curve: VoltWattCurve | None = None,
        renewables_bus: str = "renewables",
        load_bus: str = "load",
    ) -> None:
        self.network = DistributionNetwork(network_cfg)
        self.curve = curve or VoltWattCurve()
        self.renewables_bus = renewables_bus
        self.load_bus = load_bus

    def apply(self, command: DispatchCommand, observation: dict) -> tuple[DispatchCommand, VoltageReport]:
        renewable_available_kw = float(
            observation["pv_available_kw"] + observation["wind_available_kw"]
        )
        electrical_load_kw = float(observation["electrical_load_kw"])

        solution = self._solve(renewable_available_kw, electrical_load_kw)

        if not solution.converged:
            # A network solve that fails to converge is not evidence the
            # station is safe -- assume the worst (full curtailment) rather
            # than pass an unverified voltage through as if it were fine.
            limited = renewable_available_kw * self.curve.p2_frac
            return (
                replace(command, renewable_limit_kw=limited),
                VoltageReport(
                    bus_voltage_pu=solution.bus_voltage_pu,
                    converged=False,
                    curtailed=True,
                    renewable_available_kw=renewable_available_kw,
                    renewable_limit_kw=limited,
                ),
            )

        v_pu = solution.voltage_pu(self.renewables_bus)
        frac = self.curve.power_limit_frac(v_pu)
        curtailed = frac < 1.0
        limit_kw = renewable_available_kw * frac if curtailed else None

        new_command = replace(command, renewable_limit_kw=limit_kw) if curtailed else command
        return (
            new_command,
            VoltageReport(
                bus_voltage_pu=solution.bus_voltage_pu,
                converged=True,
                curtailed=curtailed,
                renewable_available_kw=renewable_available_kw,
                renewable_limit_kw=limit_kw,
            ),
        )

    def _solve(self, renewable_available_kw: float, electrical_load_kw: float) -> VoltageSolution:
        return self.network.solve(
            {
                self.renewables_bus: renewable_available_kw,
                self.load_bus: -electrical_load_kw,
            }
        )


def build_inverter_layer(cfg: StationConfig) -> InverterVoltageLayer | None:
    """The layer for `cfg`, or None if the station has no network model.

    The single place a caller should ask "does this station get Volt-Watt
    curtailment" -- `allotrope.api.simulation` and `allotrope.evaluate` both
    use this rather than checking `cfg.network is not None` themselves.
    """
    if cfg.network is None:
        return None
    return InverterVoltageLayer(cfg.network)


__all__ = ["VoltWattCurve", "VoltageReport", "InverterVoltageLayer", "build_inverter_layer"]
