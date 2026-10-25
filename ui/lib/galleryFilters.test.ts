import { describe, expect, it } from "vitest";

import {
  GALLERY_FILTERS,
  galleryFilterCounts,
  isLabelled,
  matchesGalleryFilter,
  shiftFilterCounts,
  type FilterableImage,
  type Verdict,
} from "./galleryFilters";

const box = { x0: 0, y0: 0, x1: 10, y1: 10 };

describe("isLabelled", () => {
  it("counts a background image as labelled", () => {
    expect(isLabelled({ background: true })).toBe(true);
  });

  it("trusts edited boxes over detections, even when empty", () => {
    expect(isLabelled({ editedBoxes: [], detections: [{}], detectionCount: 1 })).toBe(false);
    expect(isLabelled({ editedBoxes: [box] })).toBe(true);
  });

  it("reads the tile's counts when no detections are loaded", () => {
    expect(isLabelled({ detections: [], detectionCount: 3 })).toBe(true);
    expect(isLabelled({ labelStats: { crack: 2 } })).toBe(true);
    expect(isLabelled({ detections: [], detectionCount: 0, labelStats: {} })).toBe(false);
  });

  it("follows detections a labelling run loaded over a stale count", () => {
    expect(isLabelled({ detections: [{}], detectionCount: 0 })).toBe(true);
  });
});

describe("gallery filters", () => {
  const verdicts: Record<string, Verdict> = { b: "good", e: "bad" };
  const items: FilterableImage[] = [
    { backendId: "a", detectionCount: 2 },
    { backendId: "b", detectionCount: 1 },
    { backendId: "c", detectionCount: 0 },
    { backendId: "d", background: true },
    { backendId: "e" },
    { detections: [] },
  ];

  it("gives every pill the count of what its filter shows", () => {
    const counts = galleryFilterCounts(items, verdicts);
    for (const f of GALLERY_FILTERS) {
      expect(counts[f], f).toBe(items.filter((m) => matchesGalleryFilter(m, f, verdicts)).length);
    }
  });

  it("splits the images into unlabelled, unrated and rated", () => {
    expect(galleryFilterCounts(items, verdicts)).toEqual({
      all: 6, unlabelled: 3, unrated: 1, good: 1, bad: 1, unsure: 0,
    });
  });

  it("leaves images without boxes, background ones and uploads out of unrated", () => {
    expect(matchesGalleryFilter({ backendId: "x" }, "unrated", {})).toBe(false);
    expect(matchesGalleryFilter({ backendId: "x", background: true }, "unrated", {})).toBe(false);
    expect(matchesGalleryFilter({ detectionCount: 1 }, "unrated", {})).toBe(false);
    expect(matchesGalleryFilter({ backendId: "x", detectionCount: 1 }, "unrated", {})).toBe(true);
  });

  it("moves the counts by the images added or removed", () => {
    const counts = galleryFilterCounts(items, verdicts);
    const uploads: FilterableImage[] = [{ detections: [] }, { detections: [] }];
    expect(shiftFilterCounts(counts, uploads, {}, 1)).toEqual({
      all: 8, unlabelled: 5, unrated: 1, good: 1, bad: 1, unsure: 0,
    });
    expect(shiftFilterCounts(counts, items.slice(0, 2), verdicts, -1)).toEqual({
      all: 4, unlabelled: 3, unrated: 0, good: 0, bad: 1, unsure: 0,
    });
    expect(shiftFilterCounts(counts, items, verdicts, -1)).toEqual({
      all: 0, unlabelled: 0, unrated: 0, good: 0, bad: 0, unsure: 0,
    });
  });
});
