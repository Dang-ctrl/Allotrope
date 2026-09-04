import { useState } from "react";
import { api, ApiError } from "../api/client";

export function ControlBar({
  stationId,
  running,
  onChanged,
}: {
  stationId: string;
  running: boolean;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center gap-2">
      {!running ? (
        <button
          disabled={busy}
          onClick={() => run(() => api.startSimulation(stationId))}
          className="rounded border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-accent hover:bg-accent/20 disabled:opacity-50"
        >
          Start
        </button>
      ) : (
        <button
          disabled={busy}
          onClick={() => run(() => api.stopSimulation(stationId))}
          className="rounded border border-warn/40 bg-warn/10 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-warn hover:bg-warn/20 disabled:opacity-50"
        >
          Stop
        </button>
      )}
      <button
        disabled={busy || running}
        onClick={() => run(() => api.stepSimulation(stationId))}
        className="rounded border border-base-500 bg-base-700 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-ink-200 hover:bg-base-600 disabled:opacity-40"
        title={running ? "stop the auto-loop to single-step" : "advance one dispatch interval"}
      >
        Step
      </button>
      <button
        disabled={busy}
        onClick={() => run(() => api.resetSimulation(stationId))}
        className="rounded border border-base-500 bg-base-700 px-3 py-1.5 text-xs font-medium uppercase tracking-wide text-ink-200 hover:bg-base-600 disabled:opacity-50"
      >
        Reset
      </button>
      {error && <span className="text-xs text-crit">{error}</span>}
    </div>
  );
}
