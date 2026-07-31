"use client";

// Desktop-shell status bar (24px, 12px font, top border).
// Left: engine health dot (5s /api/health poll) + compute device
// label + workspace folder name. Right: SAM 3 state (4s
// /api/models/status poll, click opens Settings), a compact theme
// toggle, and the app version.

import { useEffect, useRef, useState, type ReactNode } from "react";

import { apiFetch } from "@/lib/apiFetch";
import {
  type EngineSettings,
  type ModelsStatus,
  downloadPct,
  fetchEngineSettings,
  fetchModelsStatus,
} from "@/lib/models";
import { useTheme } from "../ThemeProvider";

const VERSION = "v0.1.0-dev";

function deviceLabel(device: EngineSettings["device"] | null): string | null {
  if (device === "mps") return "Metal";
  if (device === "cuda") return "CUDA";
  if (device === "cpu") return "CPU";
  return null;
}

function folderName(path: string | undefined): string | null {
  if (!path) return null;
  const parts = path.replace(/[\\/]+$/, "").split(/[\\/]/);
  return parts[parts.length - 1] || null;
}

function sam3Label(status: ModelsStatus | null): string {
  if (!status) return "SAM 3 …";
  const m = status.models.sam3;
  const pct = downloadPct(m.download);
  if (pct !== null) return `SAM 3 downloading ${pct}%`;
  if (m.download?.status === "downloading") return "SAM 3 downloading…";
  if (!m.downloaded) return "SAM 3 not installed";
  if (m.loaded) return "SAM 3 loaded";
  return "SAM 3 ready";
}

function Segment({
  children,
  onClick,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  title?: string;
}) {
  const cls =
    "flex h-full items-center gap-1.5 px-2 hover:bg-[var(--surface-hover)] hover:text-[var(--foreground)] transition-colors";
  if (onClick) {
    return (
      <button type="button" onClick={onClick} title={title} className={cls}>
        {children}
      </button>
    );
  }
  return (
    <span title={title} className={cls}>
      {children}
    </span>
  );
}

export function StatusBar({ onOpenSettings }: { onOpenSettings: () => void }) {
  const [engineUp, setEngineUp] = useState<boolean | null>(null);
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const { theme, toggle } = useTheme();
  // Settings only change on user action (and the workspace path needs a
  // restart anyway) so one successful fetch is enough; re-fetch only
  // after the engine has been down, in case it restarted with a new
  // device/workspace.
  const settingsFresh = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const pollHealth = async () => {
      let up = false;
      try {
        const r = await apiFetch("/api/health", { cache: "no-store" });
        up = r.ok;
      } catch {
        up = false;
      }
      if (cancelled) return;
      setEngineUp(up);
      if (!up) {
        settingsFresh.current = false;
      } else if (!settingsFresh.current) {
        try {
          const s = await fetchEngineSettings();
          if (!cancelled) {
            setSettings(s);
            settingsFresh.current = true;
          }
        } catch {
          /* retry on the next tick */
        }
      }
    };
    pollHealth();
    const id = window.setInterval(pollHealth, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const pollModels = async () => {
      try {
        const s = await fetchModelsStatus();
        if (!cancelled) setModels(s);
      } catch {
        /* engine down — the health dot already says so */
      }
    };
    pollModels();
    const id = window.setInterval(pollModels, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const device = deviceLabel(settings?.device ?? null);
  const workspace = folderName(settings?.workspace);
  const isDark = theme === "dark";

  return (
    <footer className="flex h-6 shrink-0 items-stretch justify-between border-t border-[var(--border)] bg-[var(--background)] font-mono text-[11px] uppercase tracking-[0.08em] text-[var(--fg-dim)] tabular-nums select-none">
      <span className="flex items-stretch">
        <Segment title={engineUp ? "Engine running" : engineUp === null ? "Checking engine…" : "Engine unreachable"}>
          <span
            aria-hidden
            className={[
              "h-2 w-2 rounded-full",
              engineUp === null
                ? "bg-foreground/25"
                : engineUp
                ? "bg-[var(--ok)]"
                : "bg-[var(--bad)]",
            ].join(" ")}
          />
          <span>{engineUp === false ? "Engine down" : device ?? "Engine"}</span>
        </Segment>
        {workspace && (
          <Segment title={settings?.workspace}>
            <span className="max-w-[16rem] truncate">{workspace}</span>
          </Segment>
        )}
      </span>
      <span className="flex items-stretch">
        <Segment onClick={onOpenSettings} title="Open Settings">
          <span>{sam3Label(models)}</span>
        </Segment>
        <Segment
          onClick={toggle}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? (
            // Moon
            <svg
              viewBox="0 0 24 24"
              width={13}
              height={13}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
            </svg>
          ) : (
            // Sun
            <svg
              viewBox="0 0 24 24"
              width={13}
              height={13}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
          )}
        </Segment>
        <Segment title="PixelKit version">
          {/* Version keeps its literal casing (lowercase v + suffix). */}
          <span className="normal-case">{VERSION}</span>
        </Segment>
      </span>
    </footer>
  );
}
