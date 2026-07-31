"use client";

// Models side-bar pane: compact per-model rows over lib/models.ts.
// Each row shows the model label, its state (not downloaded /
// downloading NN% with a thin progress bar / ready / loaded) and small
// text-buttons wired to the existing download/load/unload calls.

import { useCallback, useEffect, useState } from "react";

import {
  type ModelName,
  type ModelsStatus,
  downloadModel,
  downloadPct,
  fetchModelsStatus,
  loadModel,
  unloadModel,
} from "@/lib/models";

export function ModelsPane() {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<ModelName | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchModelsStatus());
    } catch {
      /* engine unreachable — keep the last snapshot */
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 2500);
    return () => window.clearInterval(id);
  }, [refresh]);

  const act = async (name: ModelName, fn: () => Promise<unknown>) => {
    setErr(null);
    setBusy(name);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-8 shrink-0 items-center pl-4 pr-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-foreground/55 select-none">
          Models
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        {!status ? (
          <p className="px-4 py-1 text-[13px] text-foreground/40">Loading…</p>
        ) : (
          <>
            {(Object.keys(status.models) as ModelName[]).map((n) => {
              const m = status.models[n];
              const pct = downloadPct(m.download);
              const state = pct !== null
                ? `downloading ${pct}%`
                : m.loaded
                ? "loaded"
                : m.downloaded
                ? "ready"
                : "not downloaded";
              return (
                <div key={n} className="px-4 py-1.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="min-w-0 truncate text-[13px] text-foreground/85">
                      {m.label}
                    </span>
                    <span
                      className={[
                        "shrink-0 text-[11px]",
                        m.loaded ? "text-[var(--success)]" : "text-foreground/45",
                      ].join(" ")}
                    >
                      {state}
                    </span>
                  </div>
                  {pct !== null && (
                    <div className="mt-1 h-[3px] w-full overflow-hidden rounded-sm bg-foreground/10">
                      <div
                        className="h-full bg-foreground/60 transition-[width]"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  )}
                  {m.download?.status === "error" && (
                    <p className="mt-1 text-[11px] text-[var(--destructive)]">
                      {m.download.error}
                    </p>
                  )}
                  <div className="mt-0.5 flex items-center gap-2">
                    {!m.downloaded && pct === null && (
                      <ModelAction
                        label="Download"
                        disabled={busy === n}
                        onClick={() => act(n, () => downloadModel(n))}
                      />
                    )}
                    {m.downloaded && !m.loaded && (
                      <ModelAction
                        label="Load"
                        disabled={busy === n}
                        onClick={() => act(n, () => loadModel(n))}
                      />
                    )}
                    {m.loaded && (n === "sam3" || n === "vlm") && (
                      <ModelAction
                        label="Unload"
                        disabled={busy === n}
                        onClick={() => act(n, () => unloadModel(n))}
                      />
                    )}
                  </div>
                </div>
              );
            })}
            {err && (
              <p className="px-4 py-1 text-[11px] text-[var(--destructive)]">{err}</p>
            )}
            <p className="px-4 pt-2 text-[11px] text-foreground/35">
              {status.weightsDir} · {status.freeDiskGb} GB free
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function ModelAction({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="rounded border border-foreground/15 px-1.5 py-px text-[11px] text-foreground/70 hover:bg-foreground/[0.06] hover:text-foreground/95 disabled:opacity-40 transition-colors"
    >
      {label}
    </button>
  );
}
