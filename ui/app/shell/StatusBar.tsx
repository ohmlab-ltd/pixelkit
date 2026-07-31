"use client";

// Desktop-shell status bar (24px, 12px font, top border).
// Left: engine health dot (5s /api/health poll) + compute device
// label + workspace folder name. Right: the app version. Models are
// invisible in this build (the engine-health dot + device label are
// enough), and the theme is permanently dark, so there's no model
// segment and no theme toggle.

import { useEffect, useRef, useState, type ReactNode } from "react";

import { apiFetch } from "@/lib/apiFetch";
import { type EngineSettings, fetchEngineSettings } from "@/lib/models";

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

export function StatusBar() {
  const [engineUp, setEngineUp] = useState<boolean | null>(null);
  const [settings, setSettings] = useState<EngineSettings | null>(null);
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

  const device = deviceLabel(settings?.device ?? null);
  const workspace = folderName(settings?.workspace);

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
        <Segment title="PixelKit version">
          {/* Version keeps its literal casing (lowercase v + suffix). */}
          <span className="normal-case">{VERSION}</span>
        </Segment>
      </span>
    </footer>
  );
}
