import type { SafetyStatus } from "../api/types";
import { StatusPill } from "./StatusPill";

/** Mirrors allotrope/safety/projection.py's Intervention enum values exactly. */
const INTERVENTION_LABELS: Record<string, string> = {
  sanitised_non_finite_action: "Sanitised a NaN/invalid command",
  forced_start_for_capacity: "Forced a genset start to cover required capacity",
  blocked_stop_that_would_breach_reserve: "Blocked a stop that would breach reserve margin",
  clipped_setpoint_to_machine_limits: "Clipped a setpoint to machine limits",
  raised_setpoint_to_cover_critical_load: "Raised a setpoint to cover critical load",
  clipped_battery_to_thermal_envelope: "Clipped battery power to its thermal envelope",
  limited_charging_to_protect_critical_load: "Limited charging to protect critical load",
  clipped_discretionary_load: "Clipped discretionary (melt) load",
  shed_discretionary_load_for_critical: "Shed discretionary load to protect critical load",
  forced_start_to_protect_heating: "Forced a genset start to protect heating",
};

function label(code: string): string {
  return INTERVENTION_LABELS[code] ?? code;
}

export function SafetyPanel({ safety }: { safety: SafetyStatus }) {
  const report = safety.last_report;
  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wide text-ink-400">
          Safety projection
        </span>
        {report ? (
          <StatusPill tone={report.intervened ? "warn" : "ok"}>
            {report.intervened ? "intervened this step" : "action passed unmodified"}
          </StatusPill>
        ) : (
          <StatusPill tone="neutral">no data yet</StatusPill>
        )}
      </div>

      {report && report.intervened && (
        <ul className="mb-3 space-y-1 border-l-2 border-warn/40 pl-3 text-sm text-ink-200">
          {report.interventions.map((code) => (
            <li key={code}>{label(code)}</li>
          ))}
        </ul>
      )}

      {report && (
        <div className="mb-3 flex gap-4 text-xs text-ink-400">
          <span>
            required <span className="num text-ink-200">{report.required_capacity_kw.toFixed(0)} kW</span>
          </span>
          <span>
            committed <span className="num text-ink-200">{report.committed_capacity_kw.toFixed(0)} kW</span>
          </span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 border-t border-base-600 pt-3 sm:grid-cols-4">
        <Metric label="Steps" value={safety.steps.toString()} />
        <Metric
          label="Projection rate"
          value={`${(safety.projection_rate * 100).toFixed(1)}%`}
        />
        <Metric
          label="Fallback rate"
          value={`${(safety.fallback_rate * 100).toFixed(1)}%`}
          tone={safety.fallback_rate > 0 ? "warn" : "default"}
        />
        <Metric label="Max latency" value={`${safety.max_latency_ms.toFixed(2)} ms`} />
      </div>

      {safety.last_fallback_reason && (
        <div className="mt-3 rounded border border-crit/30 bg-crit/10 px-3 py-2 text-sm text-crit">
          Deterministic fallback active: {safety.last_fallback_reason}
        </div>
      )}

      {safety.voltage && (
        <div className="mt-3 border-t border-base-600 pt-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[11px] uppercase tracking-wide text-ink-400">
              Inverter Volt-Watt (network twin)
            </span>
            <StatusPill tone={safety.voltage.curtailed ? "warn" : "ok"}>
              {safety.voltage.curtailed ? "curtailing renewables" : "no curtailment"}
            </StatusPill>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-ink-400">
            {Object.entries(safety.voltage.bus_voltage_pu).map(([bus, pu]) => (
              <span key={bus}>
                {bus} <span className="num text-ink-200">{pu.toFixed(3)} pu</span>
              </span>
            ))}
          </div>
          {safety.voltage.curtailed && safety.voltage.renewable_limit_kw !== null && (
            <div className="mt-1 text-xs text-warn">
              Renewables limited to {safety.voltage.renewable_limit_kw.toFixed(1)} kW of{" "}
              {safety.voltage.renewable_available_kw.toFixed(1)} kW available
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "warn";
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-ink-400">{label}</div>
      <div className={`num text-sm ${tone === "warn" ? "text-warn" : "text-ink-100"}`}>{value}</div>
    </div>
  );
}
