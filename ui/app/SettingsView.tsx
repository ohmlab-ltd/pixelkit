"use client";

// Settings — replaces the SaaS ProfileView. Full-screen overlay in the
// profile slot: workspace, compute device, Hugging Face access, models.
import { useCallback, useEffect, useRef, useState } from "react";

import {
  EngineSettings,
  ModelInfo,
  ModelName,
  ModelsStatus,
  clearHfToken,
  downloadPct,
  fetchEngineSettings,
  fetchModelsStatus,
  setHfToken,
  setWorkspacePath,
} from "@/lib/models";

// Passive one-line model status: name + state (ready / downloading
// NN% with a thin progress bar / waiting for token). No buttons — the
// engine auto-downloads and auto-loads everything once the HF token
// exists. Shared with the first-run SetupWizard.
export function ModelStatusRow({
  m,
  tokenConfigured,
}: {
  m: ModelInfo;
  tokenConfigured: boolean;
}) {
  const pct = downloadPct(m.download);
  const state = m.downloaded
    ? "ready"
    : pct !== null
      ? `downloading ${pct}%`
      : m.gated && !tokenConfigured
        ? "waiting for token"
        : "waiting…";
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-sm">{m.label}</span>
        <span
          className={[
            "shrink-0 text-[12px] tabular-nums",
            m.downloaded ? "text-[var(--ok)]" : "text-[var(--fg-muted)]",
          ].join(" ")}
        >
          {state}
        </span>
      </div>
      {pct !== null && (
        <div className="mt-1.5 h-1 rounded-full bg-foreground/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-foreground/70 transition-[width]"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {m.download?.status === "error" && (
        <p className="mt-1 text-[12px] text-[var(--bad)]">{m.download.error}</p>
      )}
    </div>
  );
}

export function SettingsView({ onClose }: { onClose: () => void }) {
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [wsPath, setWsPath] = useState("");
  const [wsNote, setWsNote] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([fetchEngineSettings(), fetchModelsStatus()]);
      setSettings(s);
      setModels(m);
      setWsPath((prev) => prev || s.workspace);
    } catch {
      /* engine unreachable; keep last snapshot */
    }
  }, []);

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, 2000);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [refresh]);

  const act = async (fn: () => Promise<unknown>, okNote?: string) => {
    setErr(null);
    setNote(null);
    try {
      await fn();
      if (okNote) setNote(okNote);
      refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const deviceLabel =
    settings?.device === "cuda"
      ? "NVIDIA GPU (CUDA)"
      : settings?.device === "mps"
      ? "Apple GPU (Metal)"
      : "CPU (labelling unavailable)";

  return (
    <div className="fixed inset-0 z-[400] overflow-y-auto bg-[var(--background)]">
      <main className="mx-auto max-w-3xl px-6 py-8">
        <div className="flex items-center justify-between gap-4">
          <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
          <button
            onClick={onClose}
            className="text-sm text-foreground/60 hover:text-foreground transition-colors"
          >
            Close
          </button>
        </div>

        {(note || err) && (
          <p className={`mt-4 text-[13px] ${err ? "text-[var(--bad)]" : "text-[var(--ok)]"}`}>
            {err ?? note}
          </p>
        )}

        {/* Workspace */}
        <section className="mt-8 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-5">
          <h2 className="pk-micro">Workspace</h2>
          <p className="mt-1 text-[13px] text-foreground/55">
            Every project, image, annotation and model weight lives in this folder.
          </p>
          <div className="mt-3 flex gap-2">
            <input
              value={wsPath}
              onChange={(e) => setWsPath(e.target.value)}
              className="flex-1 rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm font-mono outline-none focus:border-[var(--line-strong)]"
            />
            <button
              onClick={() =>
                act(async () => {
                  const r = await setWorkspacePath(wsPath.trim());
                  setWsNote(
                    r.restartRequired
                      ? "Saved — restart PixelKit to switch to the new workspace."
                      : "Saved.",
                  );
                })
              }
              disabled={!wsPath.trim() || wsPath.trim() === settings?.workspace}
              className="rounded-md border border-[var(--line)] px-4 py-2 text-[13px] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] disabled:opacity-40"
            >
              Change
            </button>
          </div>
          {wsNote && <p className="mt-2 text-[12px] text-foreground/55">{wsNote}</p>}
        </section>

        {/* Compute */}
        <section className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-5">
          <h2 className="pk-micro">Compute</h2>
          <p className="mt-1 text-[13px] text-foreground/55">{deviceLabel}</p>
          {models && (
            <p className="mt-1 text-[12px] text-foreground/40">
              Weights: {models.weightsDir} · {models.freeDiskGb} GB free
            </p>
          )}
        </section>

        {/* Hugging Face */}
        <section className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-5">
          <h2 className="pk-micro">Hugging Face access</h2>
          <p className="mt-1 text-[13px] text-foreground/55">
            {settings?.hfTokenConfigured
              ? "A token is configured."
              : "Needed once to download the license-gated SAM 3 weights."}
          </p>
          <div className="mt-3 flex gap-2">
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder={settings?.hfTokenConfigured ? "Replace token (hf_...)" : "hf_..."}
              className="flex-1 rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm outline-none focus:border-[var(--line-strong)]"
            />
            <button
              onClick={() =>
                act(async () => {
                  const r = await setHfToken(token.trim());
                  setToken("");
                  if (r.sam3Access === false)
                    throw new Error(
                      "Token valid, but SAM 3 access is missing — accept the license at huggingface.co/facebook/sam3.",
                    );
                }, "Token validated and saved.")
              }
              disabled={!token.trim()}
              className="rounded-md bg-[var(--accent)] px-4 py-2 text-[13px] text-[var(--accent-contrast)] hover:brightness-105 disabled:opacity-40"
            >
              Save
            </button>
            {settings?.hfTokenConfigured && (
              <button
                onClick={() => act(() => clearHfToken(), "Token removed.")}
                className="rounded-md border border-[var(--line)] px-4 py-2 text-[13px] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]"
              >
                Remove
              </button>
            )}
          </div>
        </section>

        {/* Models — passive status only. The engine downloads and
            loads everything itself once the HF token exists; there is
            nothing for the user to click here. */}
        <section className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-5">
          <h2 className="pk-micro">Models</h2>
          <p className="mt-1 text-[13px] text-foreground/55">
            Models are built in and managed automatically.
          </p>
          <div className="mt-3 space-y-3">
            {models &&
              (Object.keys(models.models) as ModelName[]).map((n) => (
                <ModelStatusRow key={n} m={models.models[n]} tokenConfigured={models.hfTokenConfigured} />
              ))}
          </div>
        </section>
      </main>
    </div>
  );
}
