"use client";

// Settings — replaces the SaaS ProfileView. Full-screen overlay in the
// profile slot: workspace, compute device, Hugging Face access, models.
import { useCallback, useEffect, useRef, useState } from "react";

import {
  EngineSettings,
  ModelName,
  ModelsStatus,
  clearHfToken,
  downloadModel,
  downloadPct,
  fetchEngineSettings,
  fetchModelsStatus,
  loadModel,
  setHfToken,
  setWorkspacePath,
  unloadModel,
} from "@/lib/models";

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
      <main className="mx-auto max-w-3xl px-6 py-14">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-medium tracking-tight">Settings</h1>
          <button
            onClick={onClose}
            className="text-sm text-foreground/60 hover:text-foreground transition-colors"
          >
            Close
          </button>
        </div>

        {(note || err) && (
          <p className={`mt-4 text-[13px] ${err ? "text-red-500" : "text-emerald-500"}`}>
            {err ?? note}
          </p>
        )}

        {/* Workspace */}
        <section className="mt-8 rounded-2xl border border-foreground/10 p-6">
          <h2 className="text-sm font-medium">Workspace</h2>
          <p className="mt-1 text-[13px] text-foreground/55">
            Every project, image, annotation and model weight lives in this folder.
          </p>
          <div className="mt-3 flex gap-2">
            <input
              value={wsPath}
              onChange={(e) => setWsPath(e.target.value)}
              className="flex-1 rounded-full border border-foreground/15 bg-transparent px-4 py-2 text-sm font-mono outline-none focus:border-foreground/40"
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
              className="rounded-full border border-foreground/20 px-4 py-2 text-sm hover:bg-foreground/[0.05] disabled:opacity-40"
            >
              Change
            </button>
          </div>
          {wsNote && <p className="mt-2 text-[12px] text-foreground/55">{wsNote}</p>}
        </section>

        {/* Compute */}
        <section className="mt-4 rounded-2xl border border-foreground/10 p-6">
          <h2 className="text-sm font-medium">Compute</h2>
          <p className="mt-1 text-[13px] text-foreground/55">{deviceLabel}</p>
          {models && (
            <p className="mt-1 text-[12px] text-foreground/40">
              Weights: {models.weightsDir} · {models.freeDiskGb} GB free
            </p>
          )}
        </section>

        {/* Hugging Face */}
        <section className="mt-4 rounded-2xl border border-foreground/10 p-6">
          <h2 className="text-sm font-medium">Hugging Face access</h2>
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
              className="flex-1 rounded-full border border-foreground/15 bg-transparent px-4 py-2 text-sm outline-none focus:border-foreground/40"
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
              className="rounded-full bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
            >
              Save
            </button>
            {settings?.hfTokenConfigured && (
              <button
                onClick={() => act(() => clearHfToken(), "Token removed.")}
                className="rounded-full border border-foreground/20 px-4 py-2 text-sm hover:bg-foreground/[0.05]"
              >
                Remove
              </button>
            )}
          </div>
        </section>

        {/* Models */}
        <section className="mt-4 rounded-2xl border border-foreground/10 p-6">
          <h2 className="text-sm font-medium">Models</h2>
          <div className="mt-3 space-y-4">
            {models &&
              (Object.keys(models.models) as ModelName[]).map((n) => {
                const m = models.models[n];
                const pct = downloadPct(m.download);
                return (
                  <div key={n} className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">
                        {m.label}
                        {m.loaded && <span className="ml-2 text-[11px] text-emerald-500">loaded</span>}
                      </p>
                      <p className="text-[12px] text-foreground/45">
                        {m.repo} · ~{m.approxGb} GB
                        {m.download?.status === "error" && (
                          <span className="text-red-500"> — {m.download.error}</span>
                        )}
                      </p>
                      {pct !== null && (
                        <div className="mt-1.5 h-1.5 rounded-full bg-foreground/10 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-foreground/70 transition-[width]"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {!m.downloaded && pct === null && (
                        <button
                          onClick={() => act(() => downloadModel(n))}
                          className="rounded-full border border-foreground/20 px-3 py-1.5 text-[12px] hover:bg-foreground/[0.05]"
                        >
                          Download
                        </button>
                      )}
                      {pct !== null && (
                        <span className="text-[12px] text-foreground/50">{pct}%</span>
                      )}
                      {m.downloaded && !m.loaded && (
                        <button
                          onClick={() => act(() => loadModel(n), `${m.label} loading…`)}
                          className="rounded-full border border-foreground/20 px-3 py-1.5 text-[12px] hover:bg-foreground/[0.05]"
                        >
                          Load
                        </button>
                      )}
                      {m.loaded && (n === "vlm" || n === "sam3") && (
                        <button
                          onClick={() => act(() => unloadModel(n), `${m.label} unloaded.`)}
                          className="rounded-full border border-foreground/20 px-3 py-1.5 text-[12px] hover:bg-foreground/[0.05]"
                        >
                          Unload
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        </section>
      </main>
    </div>
  );
}
