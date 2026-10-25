// Label palette, shared by onboarding and the editor.
export const LABEL_COLOURS = [
  "#fb923c", // orange-400
  "#60a5fa", // blue-400
  "#a78bfa", // violet-400
  "#34d399", // emerald-400
  "#f472b6", // pink-400
  "#facc15", // yellow-400
  "#2dd4bf", // teal-400
  "#f87171", // red-400
];

// Hashes the canonical label, not its alias, so a rename keeps the colour.
export function colourForLabelStable(label: string): string {
  const key = (label || "").trim().toLowerCase();
  return LABEL_COLOURS[Math.abs(_hash(key)) % LABEL_COLOURS.length];
}

// Label -> colour for a project. A colliding label takes the free slot
// farthest from the taken ones. Assignment follows label order.
export function buildProjectLabelColourMap(
  allLabels: string[],
  /** Canonical lower-case label -> #rrggbb. An override takes no palette slot. */
  overrides?: Record<string, string> | null,
): Map<string, string> {
  const N = LABEL_COLOURS.length;
  const taken = new Set<number>();
  const out = new Map<string, string>();
  const ov = overrides ? new Map<string, string>(
    Object.entries(overrides).map(([k, v]) => [k.trim().toLowerCase(), v]),
  ) : null;
  for (const raw of allLabels) {
    const key = (raw || "").trim().toLowerCase();
    if (!key || out.has(key)) continue;
    if (ov && ov.has(key)) {
      out.set(key, ov.get(key)!);
      continue;
    }
    const pref = Math.abs(_hash(key)) % N;
    let slot = pref;
    if (taken.has(slot)) {
      // The free slot with the largest circular distance to any taken one.
      let bestSlot = -1;
      let bestMinDist = -1;
      for (let s = 0; s < N; s++) {
        if (taken.has(s)) continue;
        let minDist = N;
        for (const t of taken) {
          const raw_d = Math.abs(s - t);
          const d = Math.min(raw_d, N - raw_d);
          if (d < minDist) minDist = d;
        }
        if (minDist > bestMinDist) {
          bestMinDist = minDist;
          bestSlot = s;
        }
      }
      if (bestSlot >= 0) slot = bestSlot;
      else {
        // Palette exhausted, so the label shares a colour.
        let s = (pref + 1) % N;
        while (taken.has(s) && s !== pref) s = (s + 1) % N;
        slot = s;
      }
    }
    taken.add(slot);
    out.set(key, LABEL_COLOURS[slot]);
  }
  return out;
}

function _hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

// Black or white text for a background, by YIQ luminance against 150.
export function readableTextForBg(hex: string | null | undefined): string {
  if (!hex || typeof hex !== "string") return "#000";
  let s = hex.trim().replace(/^#/, "");
  if (s.length === 3) s = s.split("").map((c) => c + c).join("");
  if (s.length !== 6) return "#000";
  const r = parseInt(s.slice(0, 2), 16);
  const g = parseInt(s.slice(2, 4), 16);
  const b = parseInt(s.slice(4, 6), 16);
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return "#000";
  const yiq = (r * 299 + g * 587 + b * 114) / 1000;
  return yiq >= 150 ? "#000" : "#fff";
}
