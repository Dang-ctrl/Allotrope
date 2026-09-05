type Tone = "ok" | "warn" | "crit" | "neutral";

const TONE_CLASSES: Record<Tone, string> = {
  ok: "bg-accent/15 text-accent border-accent/30",
  warn: "bg-warn/15 text-warn border-warn/30",
  crit: "bg-crit/15 text-crit border-crit/30",
  neutral: "bg-base-600/40 text-ink-300 border-base-500/50",
};

export function StatusPill({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${TONE_CLASSES[tone]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {children}
    </span>
  );
}
