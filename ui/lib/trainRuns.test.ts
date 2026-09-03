import { describe, expect, it } from "vitest";

import {
  confusionShade,
  curveGeometry,
  etaFromProgressText,
  fmtRunDate,
  kb,
  pct,
  sizeLabel,
  statusLabel,
  type HistoryRow,
} from "./trainRuns";

describe("trainRuns helpers", () => {
  it("formats numbers the way the card shows them", () => {
    expect(pct(0.5072)).toBe("50.7%");
    expect(pct(null)).toBe("-");
    expect(kb(923392)).toBe("902 kB");
    expect(kb(5 * 1024 * 1024)).toBe("5.00 MB");
    expect(sizeLabel("tiny")).toBe("YOLOX-Tiny");
    expect(sizeLabel(undefined)).toBe("YOLOX-Nano");
    expect(fmtRunDate("not-a-run")).toBe("not-a-run");
    expect(fmtRunDate("20260820T155545Z")).not.toBe("20260820T155545Z");
  });

  it("labels run states with their progress", () => {
    expect(statusLabel({ status: "done" })).toBe("Done");
    expect(statusLabel({ status: "training", epochsDone: 3, totalEpochs: 58 })).toBe("Training · epoch 3/58");
    expect(statusLabel({ status: "interrupted", epochsDone: 12, totalEpochs: 58 })).toBe(
      "Interrupted at epoch 12/58",
    );
    expect(statusLabel({ status: "rejected" })).toBe("Rejected by ST Edge AI");
    expect(statusLabel({ status: "failed" })).toBe("Failed");
  });

  it("builds curve geometry from the history", () => {
    const history: HistoryRow[] = [
      { epoch: 1, phase: "float", loss: 8, ap50: null },
      { epoch: 2, phase: "float", loss: 4, ap50: 0.5 },
      { epoch: 3, phase: "qat", loss: 5, ap50: 0.45 },
    ];
    const g = curveGeometry(history, 4, 100, 50);
    expect(g).not.toBeNull();
    expect(g!.epochs).toBe(4);
    expect(g!.maxLoss).toBe(8);
    // Three loss vertices, two AP points, the QAT boundary at epoch 3.
    expect(g!.lossPath.split(" ")).toHaveLength(3);
    expect(g!.apPoints.map((p) => p.epoch)).toEqual([2, 3]);
    expect(g!.qatStartX).toBeCloseTo(4 + (2 / 3) * 92, 1);
    // Loss at its max sits at the top, AP 0.5 halfway down.
    expect(g!.lossPath.startsWith("M4.0,4.0")).toBe(true);
    expect(g!.apPoints[0].y).toBeCloseTo(4 + 42 / 2, 1);
    expect(curveGeometry([], 10)).toBeNull();
    expect(curveGeometry(null, 10)).toBeNull();
  });

  it("shades confusion cells relative to their row", () => {
    expect(confusionShade([0, 0, 0], 0)).toBe(0);
    expect(confusionShade([10, 5, 0], 10)).toBe(1);
    expect(confusionShade([10, 5, 0], 5)).toBeCloseTo(0.575, 3);
    expect(confusionShade([10, 5, 0], 0)).toBe(0);
  });

  it("reads the ETA the engine appends to the phase text", () => {
    expect(etaFromProgressText("loss 3.21 · AP50 0.412 · 12 min left")).toBe("12 min");
    expect(etaFromProgressText("QAT loss 2.9 · 45 s left")).toBe("45 s");
    expect(etaFromProgressText("loss 3.2 · 2 h 05 min left")).toBe("2 h 05 min");
    expect(etaFromProgressText("loss 3.2")).toBeNull();
    expect(etaFromProgressText(null)).toBeNull();
  });
});
