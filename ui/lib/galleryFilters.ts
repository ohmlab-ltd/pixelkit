// Dataset gallery filters. Each pill counts what its filter shows.

type GalleryFilter = "all" | "unlabelled" | "unrated" | "good" | "bad" | "unsure";
export type Verdict = "good" | "bad" | "unsure";

export const GALLERY_FILTERS: readonly GalleryFilter[] = ["all", "unlabelled", "unrated", "good", "bad", "unsure"];

// The fields of an image that the filters read.
export type FilterableImage = {
  backendId?: string;
  background?: boolean;
  editedBoxes?: readonly unknown[];
  detections?: readonly unknown[];
  detectionCount?: number;
  labelStats?: Record<string, number>;
};

// Marked background, or has boxes. Signals in order of trust: edited boxes,
// loaded detections, then the counts that came with the tile, which a
// labelling run leaves as they were.
export function isLabelled(m: FilterableImage): boolean {
  if (m.background) return true;
  if (m.editedBoxes !== undefined) return m.editedBoxes.length > 0;
  if ((m.detections?.length ?? 0) > 0) return true;
  if (typeof m.detectionCount === "number") return m.detectionCount > 0;
  return !!m.labelStats && Object.keys(m.labelStats).length > 0;
}

export function matchesGalleryFilter(
  m: FilterableImage,
  filter: GalleryFilter,
  verdicts: Record<string, Verdict>,
): boolean {
  if (filter === "all") return true;
  if (filter === "unlabelled") return !isLabelled(m);
  // Verdicts are keyed by backendId, so an image still uploading has none.
  if (!m.backendId) return false;
  const v = verdicts[m.backendId];
  // Boxes to rate and no verdict. A background image has nothing to rate.
  if (filter === "unrated") return !v && !m.background && isLabelled(m);
  return v === filter;
}

export function galleryFilterCounts(
  items: readonly FilterableImage[],
  verdicts: Record<string, Verdict>,
): Record<GalleryFilter, number> {
  const counts: Record<GalleryFilter, number> = { all: 0, unlabelled: 0, unrated: 0, good: 0, bad: 0, unsure: 0 };
  for (const m of items) {
    for (const f of GALLERY_FILTERS) {
      if (matchesGalleryFilter(m, f, verdicts)) counts[f]++;
    }
  }
  return counts;
}

// Counts moved by the images added (sign 1) or removed (sign -1).
export function shiftFilterCounts(
  counts: Record<GalleryFilter, number>,
  changed: readonly FilterableImage[],
  verdicts: Record<string, Verdict>,
  sign: 1 | -1,
): Record<GalleryFilter, number> {
  const d = galleryFilterCounts(changed, verdicts);
  const out = { ...counts };
  for (const f of GALLERY_FILTERS) out[f] = Math.max(0, counts[f] + sign * d[f]);
  return out;
}
