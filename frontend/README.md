# Allotrope frontend — Command Center

A React + TypeScript + Vite + Tailwind UI over `allotrope/api/` (see
[../docs/api.md](../docs/api.md)). It renders one thing: real simulation
state, polled from the backend. Nothing on screen is hardcoded or invented —
every number comes from a `fetch` call in `src/api/client.ts`, typed exactly
against the backend's actual response shapes in `src/api/types.ts`.

## Run it

Backend first (from the repo root):

```bash
pip install -e ".[api]"
uvicorn allotrope.api.app:app --reload
```

Then the frontend:

```bash
cd frontend
npm install
cp .env.example .env.local   # point VITE_API_BASE_URL elsewhere if needed
npm run dev
```

## What's here

- **Command Center** (`src/components/CommandCenter.tsx`) — the one screen
  this pass builds. Per selected station: power balance (load, critical
  load, genset output, renewables available, temperatures), the genset
  fleet (online/offline, load %, deposit level), storage (SOC, charge/
  discharge envelope), the safety projection's last decision (which
  interventions fired and why, in plain language — see
  `SafetyPanel.tsx`'s `INTERVENTION_LABELS`, which mirrors
  `allotrope/safety/projection.py`'s `Intervention` enum exactly), guard
  statistics (fallback/projection rate, max latency), and cumulative
  run metrics (fuel, black carbon, starts, wet-stacking, unmet water,
  critical unserved).
- **Live controls** — start/stop/reset/single-step the simulation, wired to
  the real `POST /stations/{id}/simulation/*` endpoints.
- **Station switcher** and an API health indicator (`GET /health`) in the
  header. When the backend is unreachable, the UI says so with an error
  banner — it never falls back to placeholder numbers.
- **Tests** (`npm run test`, Vitest + Testing Library): component tests
  against fixtures in `src/test/fixtures.ts` shaped exactly like real
  backend responses (captured from an actual running `uvicorn` process
  during development), plus an App-level integration test covering the
  station switcher, real telemetry rendering, and the unreachable-API error
  state. No Playwright/browser-automation tool was available in the
  environment this was built in, so there is no true end-to-end
  browser test yet — `npm run build` (type-checks + bundles) and manual
  `curl` verification against a live `uvicorn` + `vite` pair were used to
  confirm the app actually serves and calls the real API correctly; a real
  browser test is the honest gap this leaves.

## What's not here yet

Scoped deliberately, not overlooked — see the project's own priority order
in the root `README.md`:

- Only Command Center exists. Dedicated Digital Twin, Scenario Lab, RL
  Controller/Training, Model Registry, and Federated Learning screens are
  not built; the backend endpoints most of them would need
  (`/models`, `/evaluations`, a scenario-run trigger) don't exist yet either
  (`docs/api.md`, "What's deliberately not here yet").
- No routing (a single page, one station at a time) — there's exactly one
  real screen so far; adding a router is worth doing once there's a second
  one.
- No auth, matching the backend having none yet.
