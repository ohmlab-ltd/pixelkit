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
import { readProjectMeta } from "@/lib/projectMetaCache";

// Shown until the engine's /api/health reports its real version.
const VERSION_FALLBACK = "v0.2.1";

// Queued/running engine jobs, from GET /api/jobs/active (see gd/jobs.py
// Job.to_public — progress is {index, total, image, phase}).
type ActiveJob = {
  id: string;
  kind: string;
  project: string;
  status: "queued" | "running";
  progress?: { index?: number; total?: number; image?: string; phase?: string } | null;
  n_images?: number;
};

function jobVerb(kind: string): string {
  if (kind === "label_charlie") return "Labelling";
  if (kind === "purge_label") return "Removing label";
  if (kind === "augment_generate") return "Augmenting";
  const words = kind.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "Working";
}

// Percent complete, when derivable: progress.index / progress.total wins,
// falling back to n_images as the denominator. Null → indeterminate.
function jobPct(j: ActiveJob): number | null {
  const p = j.progress;
  const total =
    typeof p?.total === "number" && p.total > 0
      ? p.total
      : typeof j.n_images === "number" && j.n_images > 0
        ? j.n_images
        : null;
  const index = typeof p?.index === "number" ? p.index : null;
  if (total == null || index == null) return null;
  return Math.max(0, Math.min(100, Math.round((index / total) * 100)));
}

function jobProjectName(projectId: string): string {
  return readProjectMeta(projectId)?.name ?? projectId.slice(0, 8);
}

function deviceLabel(device: EngineSettings["device"] | null): string | null {
  if (device === "mps") return "Metal";
  if (device?.startsWith("cuda")) return device === "cuda" ? "CUDA" : `CUDA:${device.split(":")[1]}`;
  if (device === "cpu") return "CPU";
  return null;
}

type VramInfo = { usedGb: number; totalGb: number };

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
  const [version, setVersion] = useState<string | null>(null);
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  // Settings only change on user action (and the workspace path needs a
  // restart anyway) so one successful fetch is enough; re-fetch only
  // after the engine has been down, in case it restarted with a new
  // device/workspace.
  const settingsFresh = useRef(false);

  // Active-jobs poll for the persistent progress segment: 2 s while
  // anything is queued/running, 6 s when idle. A setTimeout chain (not
  // setInterval) so the cadence can flip per-response.
  const [jobs, setJobs] = useState<ActiveJob[]>([]);
  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const tick = async () => {
      let next: ActiveJob[] = [];
      try {
        const r = await apiFetch("/api/jobs/active", { cache: "no-store" });
        if (r.ok) {
          const d = (await r.json()) as { jobs?: ActiveJob[] };
          next = Array.isArray(d.jobs) ? d.jobs : [];
        }
      } catch {
        next = []; // engine down — the health dot already says so
      }
      if (cancelled) return;
      setJobs(next);
      timer = window.setTimeout(tick, next.length > 0 ? 2000 : 6000);
    };
    tick();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const pollHealth = async () => {
      let up = false;
      let ver: string | null = null;
      let mem: VramInfo | null = null;
      try {
        const r = await apiFetch("/api/health", { cache: "no-store" });
        up = r.ok;
        if (r.ok) {
          const d = (await r.json()) as { version?: string; vram?: VramInfo | null };
          if (typeof d.version === "string" && d.version) ver = d.version;
          if (d.vram && typeof d.vram.totalGb === "number") mem = d.vram;
        }
      } catch {
        up = false;
      }
      if (cancelled) return;
      setEngineUp(up);
      if (ver) setVersion(ver);
      setVram(up ? mem : null);
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
          <span>
            {engineUp === false ? "Engine down" : device ?? "Engine"}
            {engineUp !== false && vram ? ` ${vram.usedGb}/${vram.totalGb} GB` : ""}
          </span>
        </Segment>
        {jobs.length > 0 && (() => {
          const primary = jobs.find((j) => j.status === "running") ?? jobs[0];
          const pct = primary.status === "running" ? jobPct(primary) : null;
          const extra = jobs.length - 1;
          const tail =
            primary.status === "queued" ? " · queued" : pct != null ? ` · ${pct}%` : "…";
          return (
            <Segment
              title={jobs
                .map((j) => `${jobVerb(j.kind)} ${jobProjectName(j.project)} — ${j.status}`)
                .join("\n")}
            >
              <span
                className="relative h-[3px] w-16 shrink-0 overflow-hidden rounded-full bg-foreground/15"
                aria-hidden
              >
                {pct != null ? (
                  <span
                    className="absolute inset-y-0 left-0 rounded-full bg-[var(--accent)] transition-[width] duration-500 ease-out"
                    style={{ width: `${pct}%` }}
                  />
                ) : (
                  <span className="indeterminate-bar absolute inset-y-0 w-1/3 rounded-full bg-[var(--accent)]" />
                )}
              </span>
              <span className="max-w-[18rem] truncate normal-case tracking-normal">
                {jobVerb(primary.kind)} {jobProjectName(primary.project)}
                {tail}
              </span>
              {extra > 0 && <span className="shrink-0 normal-case tracking-normal">+{extra} more</span>}
            </Segment>
          );
        })()}
        {workspace && (
          <Segment title={settings?.workspace}>
            <span className="max-w-[16rem] truncate">{workspace}</span>
          </Segment>
        )}
      </span>
      <span className="flex items-stretch">
        <Segment title="PixelKit version">
          {/* Version keeps its literal casing (lowercase v + suffix). */}
          <span className="normal-case">{version ? `v${version}` : VERSION_FALLBACK}</span>
        </Segment>
      </span>
    </footer>
  );
}
