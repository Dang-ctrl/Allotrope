import type { Metrics } from "../api/types";
import { StatCard } from "./StatCard";

export function CumulativeMetrics({ metrics }: { metrics: Metrics }) {
  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
      <div className="mb-3 text-[11px] uppercase tracking-wide text-ink-400">
        Cumulative, this run
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Fuel" value={metrics.fuel_kl.toFixed(2)} unit="kL" />
        <StatCard label="Black carbon" value={metrics.black_carbon_g.toFixed(0)} unit="g" />
        <StatCard label="Genset starts" value={metrics.genset_starts.toFixed(0)} />
        <StatCard
          label="Wet-stacking"
          value={`${(metrics.wet_stacking_fraction * 100).toFixed(1)}%`}
          tone={metrics.wet_stacking_fraction > 0.3 ? "warn" : "default"}
        />
        <StatCard
          label="Renewable fraction"
          value={`${(metrics.renewable_fraction * 100).toFixed(1)}%`}
        />
        <StatCard
          label="Critical unserved"
          value={metrics.critical_unserved_kwh.toFixed(3)}
          unit="kWh"
          tone={metrics.critical_unserved_kwh > 0 ? "crit" : "default"}
        />
        <StatCard
          label="Freeze steps"
          value={metrics.freeze_violation_steps.toFixed(0)}
          tone={metrics.freeze_violation_steps > 0 ? "crit" : "default"}
        />
        <StatCard
          label="Unmet water"
          value={metrics.unmet_water_kwh.toFixed(0)}
          unit="kWh"
          tone={metrics.unmet_water_kwh > 0 ? "warn" : "default"}
        />
      </div>
    </div>
  );
}
