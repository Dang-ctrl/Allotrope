"""The electrical network twin: a single-feeder radial LV model in OpenDSS.

`allotrope.sim.plant.PolarMicrogrid` is a power-balance model -- kW in, kW
out, no notion of a bus, a line, or a volt. That is sufficient for the
capacity and reserve guarantees `allotrope.safety.projection` makes, but it
cannot see the one failure mode that lives entirely in the network:
inverter-driven overvoltage when renewable export outruns local load, or
undervoltage under a heavy feeder. This module is that missing piece --
solving one bus-voltage snapshot per step from OpenDSS, given each bus's net
real-power injection -- so `allotrope.safety.voltage.VoltWattCurve` has a
real number to act on rather than nothing.

What this is not: reactive power (VAr) is not modelled anywhere in this
project's plant, so this network is solved at a fixed, unity power factor
(`kvar=0` on every bus). Volt-VAr support -- injecting or absorbing reactive
power to hold voltage, the other half of IEEE 1547-2018's inverter response
curves -- would require the plant itself to carry a reactive-power balance,
which it does not. This module and `VoltWattCurve` implement Volt-**Watt**
only, honestly, rather than claiming a Volt-VAr curve with no reactive
power behind it to act on.
"""

from __future__ import annotations

from dataclasses import dataclass

from allotrope.config import NetworkConfig


@dataclass(frozen=True)
class VoltageSolution:
    """One power-flow snapshot: per-bus voltage, in per-unit of the network's base_kv."""

    bus_voltage_pu: dict[str, float]
    converged: bool

    def voltage_pu(self, bus_id: str) -> float:
        return self.bus_voltage_pu.get(bus_id, float("nan"))


class DistributionNetwork:
    """A single-feeder radial LV network, solved fresh each call.

    OpenDSS keeps its model as global interpreter state (there is one active
    circuit at a time, not one object per `DistributionNetwork` instance),
    so `solve` rebuilds the circuit from `cfg` on every call rather than
    mutating a persistent one. For a network this small (a handful of buses
    and lines) that costs microseconds, not milliseconds -- see
    `tests/test_network.py`'s latency check -- and it means two
    `DistributionNetwork` instances (say, Maitri and Bharati, if Bharati
    ever gets a `network:` section) can never cross-contaminate each
    other's circuit state.
    """

    def __init__(self, cfg: NetworkConfig) -> None:
        self.cfg = cfg

    def solve(self, bus_injection_kw: dict[str, float]) -> VoltageSolution:
        """Solve one snapshot power flow.

        `bus_injection_kw`: net real power at each non-source bus, positive
        for net generation/export, negative for net load. The source bus
        (`cfg.source_bus`) is not an input -- it is where the plant's
        aggregate gensets and storage connect, and OpenDSS holds it at the
        circuit's 1.0 pu reference, which is the modelling choice that makes
        sense for a grid-forming genset fleet.
        """
        import opendssdirect as dss

        cfg = self.cfg
        dss.Text.Command("clear")
        dss.Text.Command(
            f"new circuit.allotrope basekv={cfg.base_kv} pu=1.0 phases=3 bus1={cfg.source_bus}"
        )
        for line in cfg.lines:
            dss.Text.Command(
                f"new line.{line.from_bus}_{line.to_bus} "
                f"bus1={line.from_bus} bus2={line.to_bus} phases=3 "
                f"r1={line.r_ohm} x1={line.x_ohm} length=1 units=km"
            )
        for bus in cfg.buses:
            if bus.id == cfg.source_bus:
                continue
            # A positive injection (net export) is modelled as a negative
            # load: OpenDSS's `load` element is the simplest way to attach
            # a net real-power withdrawal (or, negated, injection) to a bus
            # without also standing up a generator model this project has
            # no other use for.
            kw = bus_injection_kw.get(bus.id, 0.0)
            dss.Text.Command(
                f"new load.{bus.id} bus1={bus.id} phases=3 kv={cfg.base_kv} "
                f"kw={-kw} kvar=0 model=1"
            )
        dss.Text.Command(f"set voltagebases=[{cfg.base_kv}]")
        dss.Text.Command("calcvoltagebases")
        dss.Text.Command("solve")

        converged = bool(dss.Solution.Converged())
        voltages: dict[str, float] = {}
        for bus in cfg.buses:
            dss.Circuit.SetActiveBus(bus.id)
            pu_mag_angle = dss.Bus.puVmagAngle()
            magnitudes = pu_mag_angle[0::2]
            voltages[bus.id] = sum(magnitudes) / len(magnitudes) if magnitudes else float("nan")
        return VoltageSolution(bus_voltage_pu=voltages, converged=converged)


__all__ = ["DistributionNetwork", "VoltageSolution"]
