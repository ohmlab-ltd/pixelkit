import { describe, expect, it } from "vitest";

import { canvasToPolygons } from "./MaskPainter";

// A stand-in canvas: alpha(x, y) for every pixel.
const canvas = (W: number, H: number, alpha: (x: number, y: number) => number) => {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data[(y * W + x) * 4 + 3] = alpha(x, y);
  return { width: W, height: H, getContext: () => ({ getImageData: () => ({ data }) }) } as unknown as HTMLCanvasElement;
};
const bbox = (polys: number[][][]) => {
  const xs = polys.flat().map((p) => p[0]);
  const ys = polys.flat().map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
};

describe("canvasToPolygons", () => {
  it("traces painted pixels at their true extent, and a re-open and save keeps it", () => {
    // Pixels 10..19 painted: the extent is [10, 20) on both axes.
    let polys = canvasToPolygons(canvas(40, 40, (x, y) => (x >= 10 && x < 20 && y >= 10 && y < 20 ? 255 : 0)));
    expect(bbox(polys)).toEqual([10, 10, 20, 20]);
    // Each re-open and save must keep the outline, not grow it by a pixel.
    const cover = (px: number, lo: number, hi: number) => Math.max(0, Math.min(px + 1, hi) - Math.max(px, lo));
    for (let i = 0; i < 3; i++) {
      const [x0, y0, x1, y1] = bbox(polys);
      polys = canvasToPolygons(canvas(40, 40, (x, y) => Math.round(255 * cover(x, x0, x1) * cover(y, y0, y1))));
    }
    expect(bbox(polys)).toEqual([10, 10, 20, 20]);
  });

  it("keeps a mask that touches the picture's edge inside the picture", () => {
    const polys = canvasToPolygons(canvas(10, 10, (x) => (x < 3 ? 255 : 0)));
    expect(bbox(polys)).toEqual([0, 0, 3, 10]);
  });
});
