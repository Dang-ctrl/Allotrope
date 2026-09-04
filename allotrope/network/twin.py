"""The OpenDSS network twin: bus voltages, and the Volt-VAr/Volt-Watt fallback.

Everything built in `allotrope.sim` is a power-balance model: kilowatts in,
kilowatts out, no voltage anywhere. That is sufficient for fuel, emissions and
life-support accounting, but it cannot support the inverter-level fallback the
project promises -- Volt-VAr and Volt-Watt curves act on voltage, and a
power-balance model has none to act on. This module is what completes that
promise: a minimal radial LV feeder solved in OpenDSS, off the same station
configuration everything else reads from.

The topology is deliberately small: one bus per asset group (PV, wind, each
storage pack, and three load classes) off a single genset bus held at the
station's nominal voltage. There is no public wiring diagram for either
station, so the topology and impedances are `[assumed]`, tagged as such in the
station YAML -- this twin demonstrates the *mechanism* the deck describes, not a
survey of the real installation.

Real power at each bus is injected as a signed OpenDSS `Load` (negative kW for
generation), which is standard practice for a net-load snapshot study and avoids
the PV-mode convergence quirks that plain `Generator` elements can introduce on
a weakly meshed radial LV network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import opendssdirect as dss

from allotrope.config import StationConfig

# IEEE 1547-2018 default Category III (aggressive, high-penetration) curves.
# [public] -- the standard itself; the specific setpoints are its default table.
VOLT_VAR_POINTS = [
    (0.90, 0.44),   # full reactive injection below this voltage
    (0.98, 0.0),    # deadband lower edge
    (1.02, 0.0),    # deadband upper edge
    (1.08, -0.44),  # full reactive absorption above this voltage
]
VOLT_WATT_POINTS = [
    (1.06, 1.0),   # no curtailment up to this voltage
    (1.10, 0.0),   # fully curtailed at and above this voltage
]

INVERTER_BUSES = ("pv", "wind", "bess_heated_core", "bess_exterior")


def _interp(points: list[tuple[float, float]], v: float) -> float:
    """Piecewise-linear interpolation, clamped flat beyond the curve's ends."""
    if v <= points[0][0]:
        return points[0][1]
    if v >= points[-1][0]:
        return points[-1][1]
    for (v0, y0), (v1, y1) in zip(points, points[1:]):
        if v0 <= v <= v1:
            if v1 == v0:
                return y0
            return y0 + (y1 - y0) * (v - v0) / (v1 - v0)
    return points[-1][1]


def volt_var_fraction(v_pu: float) -> float:
    """Reactive power command as a fraction of inverter rating, at this voltage.

    Positive injects (supports low voltage); negative absorbs (relieves high
    voltage). This is the Q the deterministic fallback issues -- no learning,
    no state beyond the present voltage reading.
    """
    return _interp(VOLT_VAR_POINTS, v_pu)


def volt_watt_fraction(v_pu: float) -> float:
    """Active power fraction of the inverter's present output to retain."""
    return _interp(VOLT_WATT_POINTS, v_pu)


@dataclass
class VoltageSolution:
    voltages_pu: dict[str, float]
    converged: bool


@dataclass
class FallbackResult:
    """One Volt-VAr/Volt-Watt pass: what each inverter bus was told to do."""

    voltages_pu: dict[str, float]
    reactive_kvar: dict[str, float]
    curtailment_fraction: dict[str, float]
    intervened_buses: list[str] = field(default_factory=list)


class NetworkTwin:
    """A radial LV feeder for one station, solved snapshot by snapshot.

    One instance owns one OpenDSS circuit for the lifetime of the process --
    `opendssdirect` is a singleton bound to the C++ engine, so instances are not
    meant to be interleaved; build one twin per station and reuse it.
    """

    def __init__(self, cfg: StationConfig) -> None:
        if cfg.network is None:
            raise ValueError(f"{cfg.site.id}: station has no [network] section configured")
        self.cfg = cfg
        self.net = cfg.network
        self._build_circuit()

    def _build_circuit(self) -> None:
        net = self.net
        dss.Text.Command("clear")
        dss.Text.Command(
            f"new circuit.{self.cfg.site.id} basekv={net.base_kv} pu=1.0 phases=3 "
            f"bus1=genset"
        )
        r1, x1 = net.feeder_r1_ohm_per_km, net.feeder_x1_ohm_per_km
        for name, branch in net.branches.items():
            dss.Text.Command(
                f"new line.feed_{name} bus1=genset bus2={name} phases=3 "
                f"length={branch.length_km} units=km r1={r1} x1={x1} r0={r1} x0={x1}"
            )
            # A near-zero placeholder load; overwritten every solve() call.
            dss.Text.Command(
                f"new load.load_{name} bus1={name} kv={net.base_kv} kw=0.001 kvar=0 "
                f"phases=3 conn=wye model=1"
            )
        dss.Text.Command(f"set voltagebases=[{net.base_kv}]")
        dss.Text.Command("calcvoltagebases")

    def solve(
        self, real_power_kw: dict[str, float], reactive_power_kvar: dict[str, float] | None = None
    ) -> VoltageSolution:
        """Solve one snapshot. Positive kW consumes; negative kW generates."""
        reactive_power_kvar = reactive_power_kvar or {}
        for name in self.net.branches:
            kw = real_power_kw.get(name, 0.0)
            kvar = reactive_power_kvar.get(name, 0.0)
            # OpenDSS chokes on an exact zero-kW load in some solver paths;
            # a tiny epsilon keeps the element well-posed without biasing results.
            kw = kw if abs(kw) > 1e-6 else 1e-6
            dss.Text.Command(f"edit load.load_{name} kw={kw} kvar={kvar}")

        dss.Text.Command("solve")
        converged = bool(dss.Solution.Converged())

        voltages = {}
        for name in self.net.branches:
            dss.Circuit.SetActiveBus(name)
            mags = dss.Bus.puVmagAngle()[0::2]  # [mag, angle, mag, angle, ...]
            voltages[name] = float(sum(mags) / len(mags)) if mags else float("nan")
        return VoltageSolution(voltages_pu=voltages, converged=converged)

    def apply_volt_var_volt_watt(
        self, real_power_kw: dict[str, float], inverter_rated_kva: dict[str, float]
    ) -> FallbackResult:
        """The two-stage inverter fallback: VAr support first, then Watt curtailment.

        This mirrors standard smart-inverter practice and the deck's own
        description -- "hardcoded Volt-VAr / Volt-Watt curves take over instantly
        if the AI times out." Reactive support is tried first because it does not
        cost the station any real energy; curtailment is the last resort, used
        only if VAr support alone could not bring a bus back into range.
        """
        first_pass = self.solve(real_power_kw)

        reactive_kvar = {}
        for name in INVERTER_BUSES:
            if name not in real_power_kw:
                continue
            v = first_pass.voltages_pu.get(name, 1.0)
            frac = volt_var_fraction(v)
            reactive_kvar[name] = frac * inverter_rated_kva.get(name, 0.0)

        second_pass = self.solve(real_power_kw, reactive_kvar)

        curtailment_fraction = {}
        curtailed_power_kw = dict(real_power_kw)
        intervened = []
        for name in INVERTER_BUSES:
            if name not in real_power_kw:
                continue
            v = second_pass.voltages_pu.get(name, 1.0)
            frac = volt_watt_fraction(v)
            curtailment_fraction[name] = frac
            if frac < 0.999:
                intervened.append(name)
                # Generation is negative kW; retaining `frac` of it means scaling
                # the negative value, which correctly reduces the magnitude of
                # power injected onto the feeder.
                curtailed_power_kw[name] = real_power_kw[name] * frac

        final_pass = self.solve(curtailed_power_kw, reactive_kvar)

        return FallbackResult(
            voltages_pu=final_pass.voltages_pu,
            reactive_kvar=reactive_kvar,
            curtailment_fraction=curtailment_fraction,
            intervened_buses=intervened,
        )


__all__ = [
    "NetworkTwin",
    "VoltageSolution",
    "FallbackResult",
    "volt_var_fraction",
    "volt_watt_fraction",
    "VOLT_VAR_POINTS",
    "VOLT_WATT_POINTS",
    "INVERTER_BUSES",
]
