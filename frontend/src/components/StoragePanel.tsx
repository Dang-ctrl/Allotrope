import type { Observation, StorageSpec } from "../api/types";

export function StoragePanel({
  storage,
  observation,
}: {
  storage: StorageSpec[];
  observation: Observation;
}) {
  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
      <div className="mb-3 text-[11px] uppercase tracking-wide text-ink-400">Storage</div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {storage.map((s, i) => {
          const soc = observation.battery_soc[i];
          const chargeKw = observation.battery_max_charge_kw[i];
          const dischargeKw = observation.battery_max_discharge_kw[i];
          const low = soc < 0.2;
          return (
            <div key={s.id} className="rounded border border-base-600 bg-base-900/60 p-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-ink-100">{s.id}</span>
                <span className="num text-xs text-ink-400">{s.capacity_kwh.toFixed(0)} kWh</span>
              </div>
              <div className={`num mt-2 text-lg ${low ? "text-warn" : "text-ink-100"}`}>
                {(soc * 100).toFixed(0)}
                <span className="text-xs text-ink-400"> % SOC</span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-base-600">
                <div
                  className={`h-full transition-[width] ${low ? "bg-warn" : "bg-accent"}`}
                  style={{ width: `${soc * 100}%` }}
                />
              </div>
              <div className="mt-1.5 flex justify-between text-xs text-ink-400">
                <span>charge ≤{chargeKw.toFixed(0)} kW</span>
                <span>discharge ≤{dischargeKw.toFixed(0)} kW</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
