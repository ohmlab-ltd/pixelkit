import { describe, expect, it } from "vitest";

import {
  confusionShade,
  curveGeometry,
  errorCounts,
  etaFromProgressText,
  fmtRunDate,
  isThresholdFallback,
  kb,
  pct,
  sizeLabel,
  statusLabel,
  thresholdNote,
  worstFirst,
  type HistoryRow,
  type ValPrediction,
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

  it("describes calibrated, fallback and configured thresholds, with or without the fallback flag", () => {
    expect(thresholdNote({ conf: 0.35, f1: 0.8, precision: 0.9, recall: 0.7, calibrated: true })).toBe(
      "chosen on the val split for the best F1 (0.80): precision 90.0%, recall 70.0%. Written into the labels header.",
    );
    expect(thresholdNote({ conf: 0.25, f1: 0, precision: 0, recall: 0, calibrated: false, fallback: true })).toMatch(
      /found nothing/,
    );
    // No flag, calibrated false, all zeros: the fallback.
    expect(isThresholdFallback({ conf: 0.25, f1: 0, precision: 0, recall: 0, calibrated: false, fallback: null })).toBe(true);
    // A user-pinned threshold that found things.
    expect(thresholdNote({ conf: 0.4, f1: 0.5, precision: 0.6, recall: 0.4, calibrated: false })).toBe(
      "as configured for this run.",
    );
  });

  it("handles one-epoch and missing-total histories", () => {
    const one = curveGeometry([{ epoch: 1, phase: "float", loss: 2, ap50: 0.3 }], null, 100, 50);
    expect(one).not.toBeNull();
    expect(one!.epochs).toBe(1);
    expect(one!.apPoints[0].x).toBeCloseTo(4 + 92 / 2, 1); // centred
    const past = curveGeometry(
      [
        { epoch: 1, phase: "float", loss: 2, ap50: null },
        { epoch: 5, phase: "qat", loss: 1, ap50: 0.5 },
      ],
      3,
      100,
      50,
    );
    expect(past!.epochs).toBe(5); // the history wins over a stale total
    expect(statusLabel({ status: "training" })).toBe("Training");
    expect(statusLabel({ status: "interrupted" })).toBe("Interrupted");
    expect(kb(512)).toBe("1 kB");
    expect(kb(100)).toBe("0 kB");
  });

  it("reads the ETA the engine appends to the phase text", () => {
    expect(etaFromProgressText("loss 3.21 · AP50 0.412 · 12 min left")).toBe("12 min");
    expect(etaFromProgressText("QAT loss 2.9 · 45 s left")).toBe("45 s");
    expect(etaFromProgressText("loss 3.2 · 2 h 05 min left")).toBe("2 h 05 min");
    expect(etaFromProgressText("loss 3.2")).toBeNull();
    expect(etaFromProgressText(null)).toBeNull();
  });

  it("ranks val images worst first, a miss or a wrong class over a spurious box", () => {
    const img = (image: string, missed: number, wrong: number, spurious: number): ValPrediction => ({
      project: "p",
      image,
      width: 10,
      height: 10,
      truth: [
        ...Array.from({ length: missed }, () => ({ box: [0, 0, 1, 1], class: 0, missed: true })),
        { box: [0, 0, 1, 1], class: 0, missed: false },
      ],
      predicted: [
        ...Array.from({ length: wrong }, () => ({ box: [0, 0, 1, 1], class: 1, score: 0.9, kind: "wrong" as const })),
        ...Array.from({ length: spurious }, () => ({ box: [0, 0, 1, 1], class: 0, score: 0.6, kind: "fp" as const })),
        { box: [0, 0, 1, 1], class: 0, score: 0.8, kind: "tp" as const },
      ],
    });
    expect(errorCounts(img("a", 1, 2, 3))).toEqual({ missed: 1, wrong: 2, spurious: 3 });
    const ranked = worstFirst([img("clean", 0, 0, 0), img("fp3", 0, 0, 3), img("fn2", 2, 0, 0),
                               img("wrong1", 0, 1, 0), img("fp2", 0, 0, 2)]);
    // fn2 weighs 4, fp3 3, then wrong1 and fp2 both 2 in val order, clean last.
    expect(ranked.map((p) => p.image)).toEqual(["fn2", "fp3", "wrong1", "fp2", "clean"]);
  });
});
