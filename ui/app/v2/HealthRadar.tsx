// Health-factor radar chart and factor metadata.

export const FACTOR_ORDER = ["balance", "coverage", "confidence", "uniqueness"] as const;

export const FACTOR_INFO: Record<string, { label: string; desc: string }> = {
  balance: { label: "Balance", desc: "How evenly examples are spread across your labels." },
  coverage: { label: "Coverage", desc: "Share of images with a detection or marked background." },
  confidence: { label: "Confidence", desc: "Share of detections not flagged unsure." },
  uniqueness: { label: "Uniqueness", desc: "Share of images that are not near-duplicates." },
};

// 0 red, 0.5 amber, 1 green.
export function factorColour(v: number): string {
  const c = Math.max(0, Math.min(1, v));
  return `hsl(${Math.round(c * 120)} 68% 45%)`;
}

// Visual style after github.com/rmontero/spider; no code from it.
export function FactorRadar({ values, size = 176 }: { values: { key: string; value: number }[]; size?: number }) {
  const n = Math.max(1, values.length);
  const cx = 120;
  const cy = 100;
  const R = 62;
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const ang = (i: number) => ((-90 + (360 / n) * i) * Math.PI) / 180;
  const at = (i: number, v: number): [number, number] => [cx + Math.cos(ang(i)) * R * v, cy + Math.sin(ang(i)) * R * v];
  const ring = (lvl: number) => values.map((_, i) => at(i, lvl).join(",")).join(" ");
  const data = values.map((f, i) => at(i, clamp(f.value)).join(",")).join(" ");
  // Edge positions for four axes, so long labels do not clip.
  const labelPos = [
    { x: cx, y: 12, anchor: "middle" as const },
    { x: 236, y: cy, anchor: "end" as const },
    { x: cx, y: 190, anchor: "middle" as const },
    { x: 4, y: cy, anchor: "start" as const },
  ];
  return (
    <svg viewBox="0 0 240 200" style={{ width: size, height: (size * 200) / 240 }} className="shrink-0" role="img" aria-label="Health factor radar">
      {/* grid */}
      {[0.25, 0.5, 0.75, 1].map((lvl) => (
        <polygon key={lvl} points={ring(lvl)} fill={lvl === 1 ? "rgb(var(--foreground-rgb) / 0.03)" : "none"} stroke="rgb(var(--foreground-rgb) / 0.14)" strokeWidth="1" />
      ))}
      {/* spokes */}
      {values.map((f, i) => {
        const [x, y] = at(i, 1);
        return <line key={`s-${f.key}`} x1={cx} y1={cy} x2={x} y2={y} stroke="rgb(var(--foreground-rgb) / 0.1)" strokeWidth="1" />;
      })}
      {/* data area */}
      <polygon points={data} fill="rgb(var(--accent-orange-rgb) / 0.18)" stroke="var(--accent-orange)" strokeWidth="2" strokeLinejoin="round" style={{ filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.12))" }} />
      {/* vertex dots */}
      {values.map((f, i) => {
        const [x, y] = at(i, clamp(f.value));
        return <circle key={`d-${f.key}`} cx={x} cy={y} r="4.5" fill={factorColour(f.value)} stroke="rgb(var(--surface-rgb))" strokeWidth="1.5" />;
      })}
      {/* axis labels */}
      {values.map((f, i) => {
        const p = labelPos[i] ?? labelPos[0];
        return (
          <text key={`t-${f.key}`} x={p.x} y={p.y} textAnchor={p.anchor} dominantBaseline="middle" style={{ fill: "rgb(var(--foreground-rgb) / 0.82)", fontSize: 11, fontWeight: 600 }}>
            {FACTOR_INFO[f.key]?.label ?? f.key}
          </text>
        );
      })}
    </svg>
  );
}
