import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { StationDetail } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import { ControlBar } from "./ControlBar";
import { CumulativeMetrics } from "./CumulativeMetrics";
import { GensetFleet } from "./GensetFleet";
import { PowerBalance } from "./PowerBalance";
import { SafetyPanel } from "./SafetyPanel";
import { StoragePanel } from "./StoragePanel";
import { StatusPill } from "./StatusPill";
import { TrendChart } from "./TrendChart";

const TELEMETRY_WINDOW = 120;

const POLL_MS = 1000;

export function CommandCenter({ stationId }: { stationId: string }) {
  const [detail, setDetail] = useState<StationDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api
      .getStation(stationId)
      .then((d) => !cancelled && setDetail(d))
      .catch((err) => !cancelled && setDetailError(err instanceof ApiError ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, [stationId]);

  const state = usePolling(() => api.getState(stationId), POLL_MS, [stationId, reloadKey]);
  const safety = usePolling(() => api.getSafety(stationId), POLL_MS, [stationId, reloadKey]);
  const metrics = usePolling(() => api.getMetrics(stationId), POLL_MS, [stationId, reloadKey]);
  const telemetry = usePolling(
    () => api.getTelemetry(stationId, TELEMETRY_WINDOW),
    POLL_MS,
    [stationId, reloadKey],
  );

  // Restarting each poll's effect (by changing a dependency) fires an
  // immediate fetch, so a control action (start/stop/reset/step) is
  // reflected right away rather than waiting up to POLL_MS for the next tick.
  const refreshNow = useCallback(() => setReloadKey((k) => k + 1), []);

  if (detailError) {
    return <ErrorBanner message={`Could not load station ${stationId}: ${detailError}`} />;
  }
  if (!detail) {
    return <div className="p-6 text-sm text-ink-400">Loading station configuration…</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 rounded-md border border-base-600 bg-base-800/60 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <span className="text-lg font-semibold text-ink-100">{detail.name}</span>
          <StatusPill tone="neutral">SIMULATION</StatusPill>
          {state.data && (
            <StatusPill tone={state.data.running ? "ok" : "neutral"}>
              {state.data.running ? "running" : "stopped"}
            </StatusPill>
          )}
          {state.data && (
            <span className="num w-full text-xs text-ink-400 sm:w-auto">
              step {state.data.step} / {state.data.n_steps} · {state.data.timestamp}
            </span>
          )}
        </div>
        <ControlBar
          stationId={stationId}
          running={state.data?.running ?? false}
          onChanged={refreshNow}
        />
      </div>

      {state.error && <ErrorBanner message={`Live state unavailable: ${state.error.message}`} />}

      {state.data && <PowerBalance observation={state.data.observation} />}

      {state.data && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <GensetFleet gensets={detail.gensets} observation={state.data.observation} />
          <StoragePanel storage={detail.storage} observation={state.data.observation} />
        </div>
      )}

      {telemetry.data && <TrendChart telemetry={telemetry.data} />}

      {safety.data && <SafetyPanel safety={safety.data} />}

      {metrics.data && <CumulativeMetrics metrics={metrics.data} />}

      <div className="rounded-md border border-base-600 bg-base-800/40 px-4 py-2 text-xs text-ink-400">
        Controller: <span className="text-ink-200">{detail.controller.name}</span> (
        {detail.controller.type}
        {detail.controller.wrapped_agent ? ` · ${detail.controller.wrapped_agent}` : ""})
      </div>
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-crit/40 bg-crit/10 px-4 py-2 text-sm text-crit">
      {message}
    </div>
  );
}
