"use client";

// First-run setup: HF token -> models download themselves -> ready.
// Shown by the workspace shell whenever SAM3 isn't usable yet;
// skippable — manual annotation works with no models at all. There
// are NO per-model buttons: once a valid token exists the engine
// auto-downloads and auto-loads everything, this card just shows the
// passive progress and flips to "All set" when both weights are in.
import { useCallback, useEffect, useRef, useState } from "react";

import {
  ModelsStatus,
  ModelName,
  fetchModelsStatus,
  setHfToken,
} from "@/lib/models";
import { ModelStatusRow } from "./SettingsView";

export function setupNeeded(s: ModelsStatus | null): boolean {
  if (!s) return false;
  return !s.models.sam3.downloaded || !s.models.dinov2.downloaded;
}

export function SetupWizard({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [token, setToken] = useState("");
  const [tokenBusy, setTokenBusy] = useState(false);
  const [tokenError, setTokenError] = useState<string | null>(null);
  const [tokenOk, setTokenOk] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchModelsStatus());
    } catch {
      /* engine restarting — keep the last snapshot */
    }
  }, []);

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, 1500);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [refresh]);

  const submitToken = async () => {
    setTokenBusy(true);
    setTokenError(null);
    setTokenOk(null);
    try {
      const res = await setHfToken(token.trim());
      if (res.sam3Access === false) {
        setTokenError(
          "Token accepted, but it can't access facebook/sam3 yet — open the model page and accept Meta's license, then re-validate.",
        );
      } else {
        setTokenOk(`Signed in as ${res.username ?? "you"} — SAM 3 access confirmed.`);
      }
      refresh();
    } catch (e) {
      setTokenError(e instanceof Error ? e.message : "Token validation failed.");
    } finally {
      setTokenBusy(false);
    }
  };

  const s = status;
  const needsToken = !!s && !s.hfTokenConfigured && !s.models.sam3.downloaded;
  const allReady = !!s && s.models.sam3.downloaded && s.models.dinov2.downloaded;

  return (
    // Outer layer scrolls; the inner grid centres the card when it fits and
    // top-aligns it when the window is short (e.g. the 700px Electron
    // minimum) so no section is clipped off-screen.
    <div className="fixed inset-0 z-[500] overflow-y-auto bg-black/40 backdrop-blur-sm">
      <div className="grid min-h-full place-items-center p-4">
      <div className="w-full max-w-xl rounded-3xl border border-foreground/10 bg-[var(--background)] p-6 shadow-2xl">
        <h2 className="text-xl font-medium tracking-tight">Set up PixelKit</h2>
        <p className="mt-2 text-sm text-foreground/60">
          PixelKit labels images locally with SAM&nbsp;3. The weights download once into your
          workspace{s ? ` (${s.freeDiskGb} GB free)` : ""}. You can skip this and annotate
          by hand — set it up any time from Settings.
        </p>

        {needsToken && (
          <div className="mt-6 rounded-2xl border border-foreground/10 p-5">
            <h3 className="text-sm font-medium">1 · Hugging Face access</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-foreground/60">
              SAM&nbsp;3 is license-gated by Meta.{" "}
              <a
                className="underline hover:text-foreground"
                href="https://huggingface.co/facebook/sam3"
                target="_blank"
                rel="noreferrer"
              >
                Open facebook/sam3
              </a>{" "}
              and accept the license, then paste a{" "}
              <a
                className="underline hover:text-foreground"
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noreferrer"
              >
                read token
              </a>
              .
            </p>
            <div className="mt-3 flex gap-2">
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="hf_..."
                className="flex-1 rounded-full border border-foreground/15 bg-transparent px-4 py-2 text-sm outline-none focus:border-foreground/40"
              />
              <button
                onClick={submitToken}
                disabled={tokenBusy || !token.trim()}
                className="rounded-full bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
              >
                {tokenBusy ? "Checking…" : "Validate"}
              </button>
            </div>
          </div>
        )}
        {tokenError && <p className="mt-3 text-[13px] text-red-500">{tokenError}</p>}
        {tokenOk && <p className="mt-3 text-[13px] text-emerald-500">{tokenOk}</p>}

        {s && (!needsToken || s.hfTokenConfigured) && (
          <div className="mt-6 rounded-2xl border border-foreground/10 p-5">
            <h3 className="text-sm font-medium">
              {needsToken ? "2 · " : ""}Model weights
            </h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-foreground/60">
              Models are built in and managed automatically — PixelKit downloads and
              loads them itself.
            </p>
            <div className="mt-3 space-y-3">
              {(Object.keys(s.models) as ModelName[])
                .filter((n) => n !== "vlm")
                .map((n) => (
                  <ModelStatusRow
                    key={n}
                    m={s.models[n]}
                    tokenConfigured={s.hfTokenConfigured}
                  />
                ))}
            </div>
          </div>
        )}

        <div className="mt-6 flex items-center justify-between">
          <button
            onClick={onClose}
            className="text-sm text-foreground/50 hover:text-foreground transition-colors"
          >
            {allReady ? "Close" : "Skip for now"}
          </button>
          {allReady && (
            <span className="text-sm text-emerald-500">All set — labelling is live.</span>
          )}
        </div>
      </div>
      </div>
    </div>
  );
}
