import { describe, expect, it } from "vitest";

import { stripTransientBoxFlags, withMasksFrom, type EditableBox } from "./BoxEditor";

const box = (id: string, x0: number, mask?: EditableBox["mask"]): EditableBox => ({
  id, label: "apple", x0, y0: 0, x1: x0 + 10, y1: 10, score: null, ...(mask !== undefined ? { mask } : {}),
});
const polys = { polygons: [[[0, 0], [5, 0], [5, 5]]] } as EditableBox["mask"];

describe("withMasksFrom", () => {
  it("fills a mask the bulk sync stripped, for a box that has not moved", () => {
    const synced = [box("a", 0), box("b", 20)];
    const held = [box("a", 0, polys), box("b", 20, polys)];
    const out = withMasksFrom(synced, held);
    expect(out.map((b) => b.mask)).toEqual([polys, polys]);
  });

  it("gives a moved box no stale mask", () => {
    const out = withMasksFrom([box("a", 3)], [box("a", 0, polys)]);
    expect(out[0].mask).toBeUndefined();
  });

  it("keeps a box's own mask and returns the same array when nothing changes", () => {
    const own = { polygons: [[[1, 1], [2, 1], [2, 2]]] } as EditableBox["mask"];
    const boxes = [box("a", 0, own)];
    expect(withMasksFrom(boxes, [box("a", 0, polys)])).toBe(boxes);
    expect(withMasksFrom(boxes, null)).toBe(boxes);
  });

  it("survives the wire strip", () => {
    const saved = stripTransientBoxFlags({ ...box("a", 0, polys), segmenting: true, maskEdited: true });
    expect(saved).toEqual({ ...box("a", 0, polys), maskEdited: true });
  });
});
