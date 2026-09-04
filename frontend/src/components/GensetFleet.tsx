import type { GensetSpec, Observation } from "../api/types";
import { StatusPill } from "./StatusPill";

export function GensetFleet({
  gensets,
  observation,
}: {
  gensets: GensetSpec[];
  observation: Observation;
}) {
  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
      <div className="mb-3 text-[11px] uppercase tracking-wide text-ink-400">Genset fleet</div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {gensets.map((g, i) => {
          const online = observation.genset_online[i];
          const powerKw = observation.genset_power_kw[i];
          const deposit = observation.genset_deposit[i];
          const loadFrac = online ? powerKw / g.rated_kw : 0;
          return (
            <div key={g.id} className="rounded border border-base-600 bg-base-900/60 p-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-ink-100">{g.id}</span>
                <StatusPill tone={online ? "ok" : "neutral"}>
                  {online ? "online" : "offline"}
                </StatusPill>
              </div>
              <div className="num mt-2 text-lg text-ink-100">
                {powerKw.toFixed(0)}
                <span className="text-xs text-ink-400"> / {g.rated_kw.toFixed(0)} kW</span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-base-600">
                <div
                  className="h-full bg-accent transition-[width]"
                  style={{ width: `${Math.min(loadFrac * 100, 100)}%` }}
                />
              </div>
              <div className="mt-1.5 flex justify-between text-xs text-ink-400">
                <span>load {Math.round(loadFrac * 100)}%</span>
                <span className={deposit > 0.5 ? "text-warn" : ""}>
                  deposit {(deposit * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
