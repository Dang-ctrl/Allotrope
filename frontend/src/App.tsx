import { useState } from "react";
import { api } from "./api/client";
import { CommandCenter } from "./components/CommandCenter";
import { StatusPill } from "./components/StatusPill";
import { usePolling } from "./hooks/usePolling";

function App() {
  const stations = usePolling(() => api.listStations(), 5000);
  const health = usePolling(() => api.health(), 5000);
  const [selected, setSelected] = useState<string | null>(null);

  const stationIds = stations.data?.map((s) => s.id) ?? [];
  const activeId = selected ?? stationIds[0] ?? null;

  return (
    <div className="min-h-screen bg-base-950">
      <header className="border-b border-base-600 bg-base-900/80 px-6 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-bold uppercase tracking-[0.2em] text-ink-100">
              Allotrope
            </span>
            <span className="text-xs text-ink-400">Antarctic microgrid command center</span>
          </div>
          <div className="flex items-center gap-3">
            {stationIds.map((id) => (
              <button
                key={id}
                onClick={() => setSelected(id)}
                className={`rounded px-2.5 py-1 text-xs font-medium uppercase tracking-wide ${
                  id === activeId
                    ? "bg-accent/20 text-accent"
                    : "text-ink-400 hover:text-ink-200"
                }`}
              >
                {id}
              </button>
            ))}
            <StatusPill tone={health.data ? "ok" : "crit"}>
              {health.data ? `api up · ${health.data.uptime_s.toFixed(0)}s` : "api unreachable"}
            </StatusPill>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-6">
        {health.error && (
          <div className="mb-4 rounded-md border border-crit/40 bg-crit/10 px-4 py-3 text-sm text-crit">
            Cannot reach the Allotrope API. Is it running? See docs/api.md — start it with{" "}
            <code className="font-mono">uvicorn allotrope.api.app:app --reload</code>.
          </div>
        )}
        {activeId ? (
          // key={activeId} remounts CommandCenter on station change, so its
          // per-station state (detail, polls) resets naturally rather than
          // needing an effect to reset it synchronously.
          <CommandCenter key={activeId} stationId={activeId} />
        ) : (
          <div className="text-sm text-ink-400">Waiting for station list…</div>
        )}
      </main>
    </div>
  );
}

export default App;
