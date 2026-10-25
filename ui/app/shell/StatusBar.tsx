"use client";

// Shell status bar: engine health, compute device, active jobs, workspace, version.

import { useEffect, useRef, useState, type ReactNode } from "react";

import { apiFetch } from "@/lib/apiFetch";
import { type EngineSettings, fetchEngineSettings } from "@/lib/models";
import { readProjectMeta } from "@/lib/projectMetaCache";

// Shown until the engine's /api/health reports its real version.
const VERSION_FALLBACK = "v0.4.0";

// Entries of GET /api/jobs/active.
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
  if (kind === "train_yolox") return "Training";
  if (kind === "label_model") return "Labelling with model";
  const words = kind.replace(/[_-]+/g, " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "Working";
}

// Percent done, or null when unknown.
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

function Segment({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className="flex h-full items-center gap-1.5 px-2 hover:bg-[var(--surface-hover)] hover:text-[var(--foreground)] transition-colors"
    >
      {children}
    </span>
  );
}

export function StatusBar() {
  const [engineUp, setEngineUp] = useState<boolean | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  // Settings are fetched once, and again after each engine outage.
  const settingsFresh = useRef(false);

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
        next = []; // engine down
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
                .map((j) => `${jobVerb(j.kind)} ${jobProjectName(j.project)}: ${j.status}`)
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
          {/* normal-case keeps the lower-case v. */}
          <span className="normal-case">{version ? `v${version}` : VERSION_FALLBACK}</span>
        </Segment>
      </span>
    </footer>
  );
}
