"use client";

import { useMemo, useState } from "react";

type Source = "validation" | "device" | "field";

const SOURCES: { id: Source; label: string; blurb: string }[] = [
  {
    id: "validation",
    label: "Validation set",
    blurb: "Replay the held-out images through the deployed model.",
  },
  {
    id: "device",
    label: "Upload from device",
    blurb: "Drop a .jsonl of inferences captured on the STM32N6.",
  },
  {
    id: "field",
    label: "Field telemetry",
    blurb: "Pull recent inferences from a connected device.",
  },
];

// Dummy summary the placeholder shows so the UI feels populated.
const FAKE_STATS = {
  total: 248,
  truePos: 191,
  falsePos: 22,
  falseNeg: 35,
  precision: 0.897,
  recall: 0.845,
};

const FAKE_SUGGESTIONS = [
  {
    title: "Collect hard negatives for your weakest class",
    detail:
      "PixelKit will suggest the specific kinds of images that would close the gap.",
  },
  {
    title: "Tighten the confidence threshold",
    detail:
      "PixelKit can preview the precision / recall trade-off for you.",
  },
  {
    title: "Retrain with stronger augmentations",
    detail: "PixelKit will pick the augmentations most likely to help based on your failure cases.",
  },
];

export function OptimiseView({
  projectName,
  hasModel,
}: {
  projectName: string;
  hasModel: boolean;
}) {
  const [source, setSource] = useState<Source>("validation");
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);

  const stats = useMemo(() => (hasRun ? FAKE_STATS : null), [hasRun]);

  const runEval = async () => {
    setRunning(true);
    await new Promise((r) => setTimeout(r, 900));
    setRunning(false);
    setHasRun(true);
  };

  return (
    <section className="mx-auto max-w-6xl px-6 pt-12 pb-24 grid gap-12">
      <div>
        <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Optimise</div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mt-1">
          Find what the model gets wrong.
        </h2>
        <p className="mt-3 max-w-xl text-[var(--muted)]">
          Feed inference results back in. Surface false positives, missed detections, and concrete
          suggestions for the next training round. Placeholder, nothing is wired yet.
        </p>
      </div>

      {!hasModel && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/[0.05] px-4 py-3 text-sm text-amber-200">
          Train and deploy a model before optimising.
        </div>
      )}

      {/* Source */}
      <div className="rounded-xl border border-[var(--border)] p-5 grid gap-5">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)]">
            Inference source
          </div>
          <button
            onClick={runEval}
            disabled={!hasModel || running}
            className="rounded-full bg-foreground text-background px-4 py-2 text-xs font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {running ? "Running…" : hasRun ? "Re-evaluate" : "Run evaluation"}
          </button>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          {SOURCES.map((s) => {
            const active = source === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setSource(s.id)}
                className={[
                  "text-left rounded-lg border p-4 transition-colors",
                  active
                    ? "border-[var(--foreground)] bg-foreground/[0.06]"
                    : "border-[var(--border)] hover:border-zinc-500",
                ].join(" ")}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={[
                      "h-3 w-3 rounded-full border transition-colors",
                      active ? "bg-white border-[var(--foreground)]" : "border-zinc-600",
                    ].join(" ")}
                  />
                  <span className="text-sm font-medium">{s.label}</span>
                </div>
                <div className="mt-1 ml-5 text-xs text-[var(--muted)]">{s.blurb}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Stats */}
      {stats ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Inferences" value={stats.total.toString()} />
            <Stat
              label="Precision"
              value={(stats.precision * 100).toFixed(1) + "%"}
              tone="neutral"
            />
            <Stat
              label="Recall"
              value={(stats.recall * 100).toFixed(1) + "%"}
              tone="neutral"
            />
            <Stat
              label="F1"
              value={(
                ((2 * stats.precision * stats.recall) / (stats.precision + stats.recall)) *
                100
              ).toFixed(1) + "%"}
            />
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <FailureCard
              title="False positives"
              count={stats.falsePos}
              tone="bad"
              blurb="Predictions with no matching ground-truth box (IoU < 0.5)."
              ctaLabel="Show false positives"
            />
            <FailureCard
              title="Missed detections"
              count={stats.falseNeg}
              tone="warn"
              blurb="Ground-truth boxes the model didn't predict at threshold."
              ctaLabel="Show missed detections"
            />
          </div>

          <div className="rounded-xl border border-[var(--border)] p-5 grid gap-4">
            <div className="text-xs uppercase tracking-wider text-[var(--muted)]">
              Suggested next steps
            </div>
            <ul className="grid gap-3">
              {FAKE_SUGGESTIONS.map((s) => (
                <li key={s.title} className="flex gap-3 items-start">
                  <span className="mt-1 h-2 w-2 rounded-full bg-foreground/60 shrink-0" />
                  <div>
                    <div className="text-sm font-medium">{s.title}</div>
                    <div className="mt-0.5 text-xs text-[var(--muted)]">{s.detail}</div>
                  </div>
                </li>
              ))}
            </ul>
            <div className="flex gap-2">
              <button
                onClick={() =>
                  alert(
                    `Would jump back to the Label tab for ${projectName} and add the suggested hard examples.`,
                  )
                }
                className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500"
              >
                Add hard examples to label
              </button>
              <button
                onClick={() => alert("Would queue a new training run with these suggestions applied.")}
                className="rounded-full bg-foreground text-background px-5 py-2 text-sm hover:bg-zinc-200"
              >
                Queue retrain
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-[var(--border)] py-16 text-center text-sm text-[var(--muted)]">
          Pick a source and click <span className="text-[var(--foreground)]">Run evaluation</span> to populate this view.
        </div>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "neutral";
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] p-4">
      <div className="text-xs text-[var(--muted)] uppercase tracking-wider">{label}</div>
      <div className={["mt-2 text-2xl font-semibold tabular-nums", tone === "neutral" ? "text-[var(--foreground)]" : "text-[var(--foreground)]"].join(" ")}>
        {value}
      </div>
    </div>
  );
}

function FailureCard({
  title,
  count,
  tone,
  blurb,
  ctaLabel,
}: {
  title: string;
  count: number;
  tone: "bad" | "warn";
  blurb: string;
  ctaLabel: string;
}) {
  const styles = {
    bad: "border-red-500/40 bg-red-500/[0.04]",
    warn: "border-amber-500/40 bg-amber-500/[0.04]",
  }[tone];
  const valueStyles = {
    bad: "text-red-300",
    warn: "text-amber-300",
  }[tone];
  return (
    <div className={["rounded-xl border p-5", styles].join(" ")}>
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-[var(--muted)]">{title}</div>
        <div className={["text-2xl font-semibold font-mono", valueStyles].join(" ")}>{count}</div>
      </div>
      <p className="mt-2 text-sm text-[var(--muted)]">{blurb}</p>
      <button
        onClick={() => alert(`${title} drill-down is a preview.`)}
        className="mt-4 rounded-full border border-[var(--border)] px-4 py-1.5 text-xs hover:border-zinc-500"
      >
        {ctaLabel}
      </button>
    </div>
  );
}
