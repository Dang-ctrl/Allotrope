export function StatCard({
  label,
  value,
  unit,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  tone?: "default" | "warn" | "crit";
}) {
  const valueColor =
    tone === "crit" ? "text-crit" : tone === "warn" ? "text-warn" : "text-ink-100";
  return (
    <div className="rounded-md border border-base-600 bg-base-800/60 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-ink-400">{label}</div>
      <div className={`num mt-1 text-2xl font-semibold ${valueColor}`}>
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-ink-400">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-xs text-ink-400">{hint}</div>}
    </div>
  );
}
