/**
 * Thin fetch wrapper over allotrope/api/app.py. Every function here maps to
 * exactly one real endpoint -- no client-side mocking, no synthetic fallback
 * data if a request fails. A failed request surfaces as a thrown error the
 * UI must show, not paper over with an invented number.
 */

import type {
  HealthStatus,
  Metrics,
  SafetyStatus,
  StationDetail,
  StationState,
  StationSummary,
  TelemetryRecord,
  ControllerStatus,
} from "./types";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

// The backend's simulation-control endpoints (start/stop/reset/step) now
// require an X-API-Key header -- see allotrope/api/app.py's module
// docstring for why. This is a credential the frontend holds, the same way
// any single-page app holds a bearer token; the *enforcement* is entirely
// server-side (FastAPI rejects the request), so this does not make the
// frontend the security boundary -- it only means a locally-run demo needs
// its operator to configure the same key on both sides. There is
// deliberately no default here: an unset key just means the four control
// buttons get a 401 until VITE_API_KEY is set to match the backend's
// ALLOTROPE_API_KEY (or the key the backend logged at startup).
const API_KEY: string | undefined = import.meta.env.VITE_API_KEY as string | undefined;

export class ApiError extends Error {
  status: number;
  path: string;

  constructor(status: number, path: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
    },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, path, body || res.statusText);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  listStations: () => request<StationSummary[]>("/stations"),
  getStation: (id: string) => request<StationDetail>(`/stations/${id}`),
  getState: (id: string) => request<StationState>(`/stations/${id}/state`),
  getTelemetry: (id: string, last?: number) =>
    request<TelemetryRecord[]>(`/stations/${id}/telemetry${last ? `?last=${last}` : ""}`),
  getMetrics: (id: string) => request<Metrics>(`/stations/${id}/metrics`),
  getSafety: (id: string) => request<SafetyStatus>(`/stations/${id}/safety`),
  getController: (id: string) => request<ControllerStatus>(`/stations/${id}/controller`),
  startSimulation: (id: string, intervalS = 0.25) =>
    request<StationState>(`/stations/${id}/simulation/start`, {
      method: "POST",
      body: JSON.stringify({ interval_s: intervalS }),
    }),
  stopSimulation: (id: string) =>
    request<StationState>(`/stations/${id}/simulation/stop`, { method: "POST" }),
  resetSimulation: (id: string) =>
    request<StationState>(`/stations/${id}/simulation/reset`, { method: "POST" }),
  stepSimulation: (id: string) =>
    request<StationState & { last_telemetry: TelemetryRecord | null }>(
      `/stations/${id}/simulation/step`,
      { method: "POST" },
    ),
};
