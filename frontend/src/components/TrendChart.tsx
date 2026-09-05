import { useMemo, useRef, useState } from "react";
import type { TelemetryRecord } from "../api/types";

/**
 * Categorical dark-mode hues validated against this app's card surface
 * (#10151d) with the dataviz skill's validator: all pairwise CVD Delta E
 * and normal-vision Delta E clear the floor, all three clear 3:1 contrast.
 * Deliberately distinct from the app's status colors (accent/warn/crit) --
 * those are reserved for state, never reused as arbitrary series identity.
 */
/**
 * `get` pulls each series' value out of a raw telemetry record. Electrical
 * load and genset output are top-level fields; critical_load_kw only
 * exists nested under `safety.critical_load_kw` (see api/types.ts's
 * TelemetryRecord for why) -- allotrope/safety/projection.py records it
 * into every SafetyReport unconditionally, every step, so it's always
 * present once at least one step has run.
 */
const SERIES = [
  {
    key: "electrical_load_kw",
    label: "Electrical load",
    color: "#3987e5",
    get: (r: TelemetryRecord) => Number(r.electrical_load_kw ?? 0),
  },
  {
    key: "genset_kw",
    label: "Genset output",
    color: "#d95926",
    get: (r: TelemetryRecord) => Number(r.genset_kw ?? 0),
  },
  {
    key: "critical_load_kw",
    label: "Critical load",
    color: "#199e70",
    get: (r: TelemetryRecord) =>
      Number((r.safety as Record<string, unknown> | null)?.critical_load_kw ?? 0),
  },
] as const;

const WIDTH = 720;
const HEIGHT = 220;
const PAD = { top: 12, right: 12, bottom: 22, left: 44 };

function niceTicks(min: number, max: number, count = 4): number[] {
  if (min === max) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (span / count) / step;
  const niceStep = err >= 7.5 ? 10 * step : err >= 3.5 ? 5 * step : err >= 1.5 ? 2 * step : step;
  const start = Math.ceil(min / niceStep) * niceStep;
  const ticks: number[] = [];
  for (let v = start; v <= max; v += niceStep) ticks.push(Math.round(v * 1000) / 1000);
  return ticks;
}

/**
 * A 2px-line trend chart of recent telemetry -- the historical-context gap
 * the rest of the Command Center leaves open (every other panel is an
 * instantaneous snapshot). Built to the dataviz skill's marks/interaction
 * specs: fixed 2px lines, hairline recessive gridlines, a legend (three
 * series), and a crosshair + one-tooltip-for-every-series hover layer
 * rather than per-point labels.
 */
export function TrendChart({ telemetry }: { telemetry: TelemetryRecord[] }) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const { points, yMin, yMax } = useMemo(() => {
    const values = telemetry.flatMap((r) => SERIES.map((s) => s.get(r)));
    const yMin = Math.min(0, ...values);
    const yMax = Math.max(1, ...values);
    return { points: telemetry, yMin, yMax };
  }, [telemetry]);

  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;

  const xFor = (i: number) =>
    points.length <= 1 ? PAD.left : PAD.left + (i / (points.length - 1)) * plotW;
  const yFor = (v: number) =>
    PAD.top + plotH - ((v - yMin) / Math.max(yMax - yMin, 1e-9)) * plotH;

  const paths = SERIES.map((s) => {
    const d = points
      .map((r, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)},${yFor(s.get(r)).toFixed(1)}`)
      .join(" ");
    return { ...s, d };
  });

  const yTicks = niceTicks(yMin, yMax);

  function handlePointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!svgRef.current || points.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const frac = Math.min(1, Math.max(0, (relX - PAD.left) / plotW));
    const idx = Math.round(frac * (points.length - 1));
    setHoverIndex(Math.min(points.length - 1, Math.max(0, idx)));
  }

  if (points.length === 0) {
    return (
      <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-400">
          Recent trend
        </div>
        <div className="py-8 text-center text-sm text-ink-400">
          No telemetry yet -- run the simulation to build history.
        </div>
      </div>
    );
  }

  const hovered = hoverIndex !== null ? points[hoverIndex] : null;
  const hoverX = hoverIndex !== null ? xFor(hoverIndex) : null;
  // Flip the tooltip to the left half once the crosshair passes the
  // midpoint, so it never runs off the right edge of the chart.
  const tooltipOnLeft = hoverX !== null && hoverX > PAD.left + plotW / 2;

  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wide text-ink-400">Recent trend</span>
        <ul className="flex items-center gap-4 text-xs text-ink-300">
          {SERIES.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block h-0.5 w-3 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
            </li>
          ))}
        </ul>
      </div>

      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full touch-none"
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHoverIndex(null)}
          role="img"
          aria-label="Recent electrical load, genset output, and critical load over time"
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={yFor(t)}
                y2={yFor(t)}
                stroke="#232b38"
                strokeWidth={1}
              />
              <text x={PAD.left - 8} y={yFor(t)} textAnchor="end" dominantBaseline="middle" className="fill-ink-400" fontSize={10}>
                {Math.round(t)}
              </text>
            </g>
          ))}

          {paths.map((p) => (
            <path key={p.key} d={p.d} fill="none" stroke={p.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}

          {hoverX !== null && (
            <line x1={hoverX} x2={hoverX} y1={PAD.top} y2={PAD.top + plotH} stroke="#3a4657" strokeWidth={1} />
          )}
          {hoverIndex !== null &&
            SERIES.map((s) => (
              <circle
                key={s.key}
                cx={hoverX!}
                cy={yFor(s.get(points[hoverIndex]))}
                r={4}
                fill={s.color}
                stroke="#10151d"
                strokeWidth={2}
              />
            ))}
        </svg>

        {hovered && hoverX !== null && (
          <div
            className="pointer-events-none absolute top-0 min-w-[160px] rounded border border-base-500 bg-base-900/95 px-3 py-2 text-xs shadow-lg"
            style={{
              left: tooltipOnLeft ? undefined : `${(hoverX / WIDTH) * 100}%`,
              right: tooltipOnLeft ? `${100 - (hoverX / WIDTH) * 100}%` : undefined,
              transform: tooltipOnLeft ? "translateX(-8px)" : "translateX(8px)",
            }}
          >
            <div className="num mb-1 text-ink-300">{String(hovered.timestamp)}</div>
            {SERIES.map((s) => (
              <div key={s.key} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-ink-300">
                  <span aria-hidden className="inline-block h-0.5 w-3 rounded-full" style={{ backgroundColor: s.color }} />
                  {s.label}
                </span>
                <span className="num font-semibold text-ink-100">
                  {s.get(hovered).toFixed(1)} kW
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
