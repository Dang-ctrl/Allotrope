"""Conversions between the project's dataclasses and their wire types.

Kept in one place so the server and client cannot drift from each other on
field order or units -- everything here is kilowatts, litres and milligrams,
the same units used throughout `allotrope.sim`.
"""

from __future__ import annotations

from allotrope.rpc import allotrope_pb2 as pb
from allotrope.safety.projection import SafetyReport
from allotrope.sim.plant import DispatchCommand


def command_to_proto(command: DispatchCommand) -> pb.DispatchRequest:
    return pb.DispatchRequest(
        genset_on=list(command.genset_on),
        genset_setpoint_kw=[float(v) for v in command.genset_setpoint_kw],
        battery_kw=[float(v) for v in command.battery_kw],
        snow_melt_kw=float(command.snow_melt_kw),
    )


def command_from_proto(request: pb.DispatchRequest) -> DispatchCommand:
    return DispatchCommand(
        genset_on=tuple(bool(v) for v in request.genset_on),
        genset_setpoint_kw=tuple(float(v) for v in request.genset_setpoint_kw),
        battery_kw=tuple(float(v) for v in request.battery_kw),
        snow_melt_kw=float(request.snow_melt_kw),
    )


def telemetry_to_proto(t: dict) -> pb.Telemetry:
    return pb.Telemetry(
        genset_kw=t["genset_kw"],
        genset_power_kw=list(t["genset_power_kw"]),
        genset_online=list(t["genset_online"]),
        fuel_l=t["fuel_l"],
        black_carbon_mg=t["black_carbon_mg"],
        renewable_used_kw=t["renewable_used_kw"],
        curtailed_kw=t["curtailed_kw"],
        battery_kw=list(t["battery_kw"]),
        battery_soc=list(t["battery_soc"]),
        electrical_load_kw=t["electrical_load_kw"],
        melt_kw=t["melt_kw"],
        unserved_kw=t["unserved_kw"],
        critical_unserved_kw=t["critical_unserved_kw"],
        indoor_temp_c=t["indoor_temp_c"],
        air_temp_c=t["air_temp_c"],
    )


def safety_report_to_proto(report: SafetyReport) -> pb.SafetyReport:
    return pb.SafetyReport(
        intervened=report.intervened,
        interventions=[i.value for i in report.interventions],
        required_capacity_kw=report.required_capacity_kw,
        committed_capacity_kw=report.committed_capacity_kw,
    )


def observation_to_proto(obs: dict) -> pb.Observation:
    return pb.Observation(
        electrical_load_kw=obs["electrical_load_kw"],
        critical_load_kw=obs["critical_load_kw"],
        firm_thermal_kw=obs["firm_thermal_kw"],
        pv_available_kw=obs["pv_available_kw"],
        wind_available_kw=obs["wind_available_kw"],
        air_temp_c=obs["air_temp_c"],
        wind_speed_ms=obs["wind_speed_ms"],
        indoor_temp_c=obs["indoor_temp_c"],
        genset_online=list(obs["genset_online"]),
        genset_power_kw=list(obs["genset_power_kw"]),
        genset_deposit=list(obs["genset_deposit"]),
        battery_soc=list(obs["battery_soc"]),
    )


__all__ = [
    "command_to_proto",
    "command_from_proto",
    "telemetry_to_proto",
    "safety_report_to_proto",
    "observation_to_proto",
]
