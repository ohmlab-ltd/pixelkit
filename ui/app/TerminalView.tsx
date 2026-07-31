"use client";

import { useEffect, useMemo, useState } from "react";
import { Footer } from "./Footer";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "pixelkit.terminalToken";
const ALLOWED_USERS = ["hamish", "mukund"];

// ── Types ─────────────────────────────────────────────────────────────────────

type Job = {
  id: string;
  kind: "label" | "label_lite" | "segment" | "segment_box" | "classify_box" | "detect_point" | "upload" | "nsfw_check" | "train";
  project: string;
  user: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";
  queuedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progress: { index?: number; total?: number; image?: string; phase?: string };
  error: string | null;
  elapsedS: number;
  n_images: number;
  costPence: number;
};

type Totals = {
  projects: number;
  images_total: number;
  images_labelled: number;
  nsfw_blocks_total: number;
  signups_total: number;
};

type LiveUsers = {
  count: number;
  users: string[];
};

type SignupEvent = {
  ts: string;
  kind: "signup";
  username?: string;
  name?: string;
  email?: string;
  image?: string;
  provider?: string;
};

type Stats = {
  running: number;
  queued: number;
  todayCount: number;
  todayCostPence: number;
  powerW: number;
  costPencePerKwh: number;
};

type Event = {
  ts: string;
  kind: string;
  project?: string;
  file?: string;
  score?: number;
  classification?: string;
};

type DemoEvent = {
  ts: string;
  kind: "demo";
  ip?: string;
  country?: string;
  status?: string;
  mode?: string;
  n_labels?: number;
  n_test?: number;
  n_refs?: number;
  n_detections?: number;
};

type SystemEvent = {
  ts: string;
  kind: "system";
  action?: string;
  model?: string;
  project?: string;
  vram_mb?: number;
  before_mb?: number;
  after_mb?: number;
  freed_mb?: number;
  error?: string;
  n_train?: number;
  n_val?: number;
  n_classes?: number;
  imgsz?: number;
  epochs?: number;
  batch?: number;
  best_val_loss?: number;
};

type UserRecord = {
  user: string;
  projects: {
    id: string;
    name: string;
    n_images: number;
    n_labelled: number;
    tags: string[];
    updated: string;
  }[];
};

type Filter = "all" | "active" | "today" | "failed";

// ── Constants ─────────────────────────────────────────────────────────────────

const KIND_LABEL: Record<Job["kind"], string> = {
  label: "Label",
  label_lite: "Lite re-run",
  segment: "Segment",
  segment_box: "Box mask",
  classify_box: "Box label",
  detect_point: "Click",
  upload: "Upload",
  nsfw_check: "NSFW",
  train: "Train",
};

// ── TerminalView ──────────────────────────────────────────────────────────────

export function TerminalView({ username }: { username: string }) {
  const allowed = ALLOWED_USERS.includes(username);
  const [token, setToken] = useState<string | null>(null);

  // Priority data (fast poll)
  const [jobs, setJobs] = useState<Job[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [live, setLive] = useState<LiveUsers>({ count: 0, users: [] });

  // Secondary data (slow poll, lazy)
  const [events, setEvents] = useState<Event[]>([]);
  const [signups, setSignups] = useState<SignupEvent[]>([]);
  const [systemEvents, setSystemEvents] = useState<SystemEvent[]>([]);
  const [demoEvents, setDemoEvents] = useState<DemoEvent[]>([]);
  const [totals, setTotals] = useState<Totals | null>(null);
  const [userRecords, setUserRecords] = useState<UserRecord[]>([]);

  const [filter, setFilter] = useState<Filter>("all");
  const [error, setError] = useState<string | null>(null);
  const [secondaryReady, setSecondaryReady] = useState(false);

  // Hydrate token from localStorage.
  useEffect(() => {
    if (!allowed) return;
    const stored = typeof window !== "undefined" ? localStorage.getItem(TOKEN_KEY) : null;
    if (!stored) return;
    (async () => {
      try {
        const r = await fetch(`${API}/api/terminal/whoami`, {
          headers: { "X-Terminal-Token": stored },
        });
        if (r.ok) setToken(stored);
        else localStorage.removeItem(TOKEN_KEY);
      } catch {
        setToken(stored);
      }
    })();
  }, [allowed]);

  // Priority poll: stats + jobs + live — every 1s.
  useEffect(() => {
    if (!allowed || !token) return;
    let alive = true;
    const headers = { "X-Terminal-Token": token };
    const tick = async () => {
      try {
        const [jr, sr, lr] = await Promise.all([
          fetch(`${API}/api/jobs`, { cache: "no-store", headers }),
          fetch(`${API}/api/jobs/stats`, { cache: "no-store", headers }),
          fetch(`${API}/api/users/live`, { cache: "no-store", headers }),
        ]);
        if (jr.status === 401 || sr.status === 401) {
          if (!alive) return;
          localStorage.removeItem(TOKEN_KEY);
          setToken(null);
          setError("Session expired, re-enter the token.");
          return;
        }
        if (!jr.ok || !sr.ok) throw new Error(`http ${jr.status}/${sr.status}`);
        if (!alive) return;
        setJobs(await jr.json());
        setStats(await sr.json());
        if (lr.ok) setLive(await lr.json());
        setError(null);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => { alive = false; window.clearInterval(id); };
  }, [token, allowed]);

  // Secondary poll: events + totals + user records — every 10s, delayed 1.5s.
  useEffect(() => {
    if (!allowed || !token) return;
    let alive = true;
    const headers = { "X-Terminal-Token": token };
    const tick = async () => {
      try {
        const [er, tr, gr, sysr, dr, ur] = await Promise.all([
          fetch(`${API}/api/events?kind=nsfw_block&limit=100`, { cache: "no-store", headers }),
          fetch(`${API}/api/stats/totals`, { cache: "no-store", headers }),
          fetch(`${API}/api/events?kind=signup&limit=50`, { cache: "no-store", headers }),
          fetch(`${API}/api/events?kind=system&limit=200`, { cache: "no-store", headers }),
          fetch(`${API}/api/events?kind=demo&limit=100`, { cache: "no-store", headers }),
          fetch(`${API}/api/terminal/users`, { cache: "no-store", headers }),
        ]);
        if (!alive) return;
        if (er.ok) setEvents(await er.json());
        if (tr.ok) setTotals(await tr.json());
        if (gr.ok) setSignups(await gr.json());
        if (sysr.ok) setSystemEvents(await sysr.json());
        if (dr.ok) setDemoEvents(await dr.json());
        if (ur.ok) setUserRecords(await ur.json());
        setSecondaryReady(true);
      } catch {
        // non-fatal
      }
    };
    let intervalId: number | null = null;
    const delay = window.setTimeout(() => {
      tick();
      intervalId = window.setInterval(tick, 10000) as unknown as number;
    }, 1500);
    return () => {
      alive = false;
      window.clearTimeout(delay);
      if (intervalId !== null) window.clearInterval(intervalId);
    };
  }, [token, allowed]);

  // All hooks run unconditionally.
  const filtered = useMemo(() => {
    if (filter === "active") return jobs.filter((j) => j.status === "running" || j.status === "queued");
    if (filter === "failed") return jobs.filter((j) => j.status === "failed");
    if (filter === "today") {
      const today = new Date().toISOString().slice(0, 10);
      return jobs.filter((j) => j.queuedAt.startsWith(today));
    }
    return jobs;
  }, [jobs, filter]);

  // Sparkline data: jobs per 2-hour bucket over the last 24h.
  const activityBuckets = useMemo(() => {
    const now = Date.now();
    const BUCKETS = 12;
    const BUCKET_MS = 2 * 60 * 60 * 1000;
    const counts = new Array(BUCKETS).fill(0);
    for (const j of jobs) {
      const t = new Date(j.queuedAt).getTime();
      const age = now - t;
      if (age < 0 || age > BUCKETS * BUCKET_MS) continue;
      const bucket = Math.floor(age / BUCKET_MS);
      counts[BUCKETS - 1 - bucket]++;
    }
    return counts;
  }, [jobs]);

  if (!allowed) return <Forbidden />;

  if (!token) {
    return (
      <LockScreen
        onUnlock={async (input) => {
          try {
            const r = await fetch(`${API}/api/terminal/whoami`, {
              headers: { "X-Terminal-Token": input },
            });
            if (!r.ok) return "Invalid token";
            localStorage.setItem(TOKEN_KEY, input);
            setToken(input);
            return null;
          } catch (e) {
            return e instanceof Error ? e.message : String(e);
          }
        }}
      />
    );
  }

  const cancel = async (id: string) => {
    try {
      await fetch(`${API}/api/jobs/${id}`, {
        method: "DELETE",
        headers: { "X-Terminal-Token": token },
      });
    } catch { /* surface via polling */ }
  };

  const signOut = () => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
  };

  return (
    <main className="min-h-screen bg-[var(--background)]">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pt-12 pb-6">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <div className="text-xs text-orange-300 tracking-widest uppercase font-mono">Terminal</div>
            <h1 className="mt-1 text-4xl md:text-5xl font-light tracking-tight">Pixel Kit Admin</h1>
            <p className="mt-2 max-w-2xl text-foreground/45 text-sm">
              Live backend activity. Cost assumes{" "}
              <span className="text-foreground/80">$0.207 / GPU-hour</span>.
              Polling: priority 1s · secondary 10s.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {error && <div className="text-xs text-red-300 font-mono">err: {error}</div>}
            <button
              onClick={signOut}
              className="text-xs text-foreground/60 hover:text-foreground border border-foreground/15 bg-foreground/5 hover:bg-foreground/10 rounded-full px-3 py-1.5 transition-colors font-mono"
            >
              lock
            </button>
          </div>
        </div>
      </section>

      {/* ── BigStats row ───────────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-6 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        <BigStat label="Live now" value={live.count} accent="emerald" span={1} />
        <BigStat label="Running" value={stats?.running ?? 0} accent="orange" span={1} />
        <BigStat label="Queued" value={stats?.queued ?? 0} span={1} />
        <BigStat label="Jobs today" value={stats?.todayCount ?? 0} span={1} />
        <BigStat label="Total projects" value={totals?.projects ?? "—"} span={1} />
        <BigStat label="Images labelled" value={totals?.images_labelled ?? "—"} span={1} />
        <BigStat label="Signups" value={totals?.signups_total ?? "—"} span={1} />
        <BigStat label="Spend today" value={formatCost(stats?.todayCostPence ?? 0)} accent="amber" span={1} />
      </section>

      {/* ── Job activity chart ─────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-6 grid md:grid-cols-[1fr_auto] gap-4 items-start">
        <PanelCard>
          <PanelHeader title="Job activity" subtitle="2-hour buckets · last 24h" />
          <div className="px-5 py-4">
            <Sparkline buckets={activityBuckets} height={48} />
            <div className="mt-1 flex justify-between text-[10px] font-mono text-foreground/30">
              <span>24h ago</span>
              <span>now</span>
            </div>
          </div>
        </PanelCard>
        <div className="hidden md:flex flex-col gap-3 min-w-[160px]">
          <MetricPill label="NSFW blocks" value={totals?.nsfw_blocks_total ?? "—"} />
          <MetricPill label="Demo runs" value={demoEvents.length} />
        </div>
      </section>

      {/* ── Job list ───────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-4 flex items-center gap-2 flex-wrap">
        <span className="text-[10px] uppercase font-mono text-foreground/40 tracking-widest mr-1">Filter</span>
        {(["active", "today", "failed", "all"] as Filter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={[
              "px-3 py-1 rounded-full text-[10px] uppercase tracking-widest font-mono border transition-colors",
              filter === f
                ? "bg-foreground text-background border-foreground"
                : "border-foreground/15 text-foreground/50 hover:text-foreground hover:border-foreground/30",
            ].join(" ")}
          >
            {f}
          </button>
        ))}
        <span className="ml-auto text-[10px] font-mono text-foreground/35 tabular-nums">
          {filtered.length} jobs
        </span>
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-8">
        <PanelCard>
          <div className="grid grid-cols-[5.5rem_5rem_3.5rem_1fr_7rem_5rem_5rem_5rem_4rem_2rem] gap-3 px-5 py-2 text-[9px] uppercase tracking-widest text-foreground/35 border-b border-foreground/8 font-mono">
            <span>Status</span>
            <span>Kind</span>
            <span className="text-right">N</span>
            <span>Project · Image</span>
            <span>User</span>
            <span>ID</span>
            <span className="text-right">Queued</span>
            <span className="text-right">Elapsed</span>
            <span className="text-right">Cost</span>
            <span />
          </div>
          {filtered.length === 0 ? (
            <div className="px-5 py-8 text-center text-xs text-foreground/35 font-mono">no jobs match.</div>
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-[26rem] overflow-y-auto">
              {filtered.map((j) => (
                <JobRow key={j.id} job={j} onCancel={() => cancel(j.id)} />
              ))}
            </ul>
          )}
        </PanelCard>
      </section>

      {/* ── Three-panel row: live / signups / NSFW ─────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-6 grid md:grid-cols-3 gap-4">
        <PanelCard>
          <PanelHeader title="Live now" count={live.count} />
          {live.users.length === 0 ? (
            <EmptyState text="no active users." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-72 overflow-y-auto">
              {live.users.map((u) => (
                <li key={u} className="px-5 py-2 flex items-center gap-2 text-xs">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shrink-0" />
                  <span className="truncate font-mono text-foreground/80">@{u}</span>
                </li>
              ))}
            </ul>
          )}
        </PanelCard>

        <PanelCard>
          <PanelHeader title="Recent signups" count={secondaryReady ? signups.length : undefined} />
          {!secondaryReady ? (
            <EmptyState text="loading…" />
          ) : signups.length === 0 ? (
            <EmptyState text="no signups yet." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-72 overflow-y-auto">
              {signups.map((u, i) => (
                <li key={i} className="px-5 py-2 flex items-center gap-3 text-xs">
                  {u.image ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={u.image} alt="" className="h-6 w-6 rounded-full object-cover shrink-0" referrerPolicy="no-referrer" />
                  ) : (
                    <span className="h-6 w-6 rounded-full bg-foreground/10 grid place-items-center text-[9px] shrink-0 font-mono">
                      {(u.name || u.username || u.email || "?")[0]?.toUpperCase()}
                    </span>
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-foreground/85">{u.name || u.username || u.email}</div>
                    <div className="text-foreground/35 text-[10px] truncate font-mono">
                      {u.provider ?? "—"} · {u.email ?? "—"}
                    </div>
                  </div>
                  <DualTime iso={u.ts} />
                </li>
              ))}
            </ul>
          )}
        </PanelCard>

        <PanelCard>
          <PanelHeader title="NSFW blocks" count={secondaryReady ? events.length : undefined} />
          {!secondaryReady ? (
            <EmptyState text="loading…" />
          ) : events.length === 0 ? (
            <EmptyState text="no blocks recorded." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-72 overflow-y-auto">
              {events.map((e, i) => (
                <li key={i} className="px-5 py-2 flex items-start gap-3 text-xs">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-foreground/80 font-mono">{e.project}/{e.file}</div>
                    <div className="text-foreground/35 text-[10px] truncate">
                      {e.classification ?? "—"} · score {e.score?.toFixed(3)} · {(e as { user?: string }).user ?? "—"}
                    </div>
                  </div>
                  <DualTime iso={e.ts} />
                </li>
              ))}
            </ul>
          )}
        </PanelCard>
      </section>

      {/* ── Demo runs + User projects ──────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-6 grid md:grid-cols-2 gap-4">
        <PanelCard>
          <PanelHeader title="Demo runs" count={secondaryReady ? demoEvents.length : undefined} subtitle="with IPs" />
          {!secondaryReady ? (
            <EmptyState text="loading…" />
          ) : demoEvents.length === 0 ? (
            <EmptyState text="no demo runs recorded." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-80 overflow-y-auto font-mono">
              {demoEvents.map((e, i) => (
                <li key={i} className="px-5 py-2 grid grid-cols-[1fr_auto] gap-3 items-start text-xs">
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={[
                          "inline-block h-1.5 w-1.5 rounded-full shrink-0",
                          e.status === "ok" ? "bg-emerald-400" : "bg-red-400",
                        ].join(" ")}
                      />
                      <span className="text-foreground/80">{e.ip ?? "—"}</span>
                      {e.country && (
                        <span className="text-foreground/40 text-[10px]">{e.country}</span>
                      )}
                    </div>
                    <div className="mt-0.5 text-foreground/35 text-[10px]">
                      {e.n_labels ?? 0} labels · {e.n_test ?? 0} test · {e.n_refs ?? 0} refs ·{" "}
                      {e.n_detections ?? 0} det
                    </div>
                  </div>
                  <DualTime iso={e.ts} />
                </li>
              ))}
            </ul>
          )}
        </PanelCard>

        <PanelCard>
          <PanelHeader title="User projects" count={secondaryReady ? userRecords.length : undefined} subtitle="private" />
          {!secondaryReady ? (
            <EmptyState text="loading…" />
          ) : userRecords.length === 0 ? (
            <EmptyState text="no projects found." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-80 overflow-y-auto">
              {userRecords.map(({ user, projects }) => (
                <li key={user} className="px-5 py-2.5">
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-xs font-mono text-foreground/85">@{user}</span>
                    <span className="text-[10px] font-mono text-foreground/40">{projects.length} project{projects.length !== 1 ? "s" : ""}</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {projects.map((p) => (
                      <div
                        key={p.id}
                        className="text-[10px] font-mono px-2 py-0.5 rounded border border-foreground/10 bg-foreground/[0.03] text-foreground/60"
                        title={`${p.n_images} images · ${p.n_labelled} labelled · ${p.tags.join(", ")}`}
                      >
                        {p.name.length > 20 ? p.name.slice(0, 20) + "…" : p.name}
                        <span className="text-foreground/30 ml-1">{p.n_images}</span>
                      </div>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </PanelCard>
      </section>

      {/* ── System events ─────────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 pb-10">
        <PanelCard>
          <PanelHeader title="System events" count={secondaryReady ? systemEvents.length : undefined} />
          {!secondaryReady ? (
            <EmptyState text="loading…" />
          ) : systemEvents.length === 0 ? (
            <EmptyState text="no system events." />
          ) : (
            <ul className="divide-y divide-foreground/5 max-h-80 overflow-y-auto font-mono">
              {systemEvents.map((e, i) => (
                <SystemEventRow key={i} event={e} />
              ))}
            </ul>
          )}
        </PanelCard>
      </section>

      <Footer />
    </main>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

const SYSTEM_ACTION_LABEL: Record<string, { text: string; tone: "neutral" | "amber" | "emerald" | "red" }> = {
  vram_clear: { text: "VRAM cleared", tone: "neutral" },
  inference_unload_start: { text: "Inference unload", tone: "amber" },
  model_unload: { text: "Model unloaded", tone: "amber" },
  model_unload_failed: { text: "Model unload failed", tone: "red" },
  model_load_start: { text: "Model loading…", tone: "neutral" },
  model_load: { text: "Model loaded", tone: "emerald" },
  model_warm_start: { text: "Model warming…", tone: "neutral" },
  model_warm: { text: "Model warm", tone: "emerald" },
  model_warm_failed: { text: "Model warm failed", tone: "red" },
  train_wait_for_queue: { text: "Train waiting", tone: "amber" },
  train_start: { text: "Train started", tone: "amber" },
  train_done: { text: "Train done", tone: "emerald" },
  train_failed: { text: "Train failed", tone: "red" },
};

function SystemEventRow({ event }: { event: SystemEvent }) {
  const action = event.action || "event";
  const meta = SYSTEM_ACTION_LABEL[action] || { text: action, tone: "neutral" as const };
  const dot =
    meta.tone === "emerald" ? "bg-emerald-400" :
    meta.tone === "amber" ? "bg-amber-300" :
    meta.tone === "red" ? "bg-red-400" :
    "bg-foreground/35";
  const detail: string[] = [];
  if (event.model) detail.push(event.model);
  if (event.project) detail.push(`p=${event.project.slice(0, 8)}…`);
  if (typeof event.vram_mb === "number") detail.push(`vram ${event.vram_mb}MB`);
  if (typeof event.freed_mb === "number" && event.freed_mb > 0) detail.push(`freed ${event.freed_mb}MB`);
  if (typeof event.epochs === "number") detail.push(`${event.epochs}ep`);
  if (typeof event.n_train === "number") detail.push(`tr${event.n_train}`);
  if (typeof event.n_val === "number") detail.push(`val${event.n_val}`);
  if (typeof event.best_val_loss === "number") detail.push(`loss${event.best_val_loss.toFixed(4)}`);
  if (event.error) detail.push(event.error.slice(0, 40));
  return (
    <li className="px-5 py-2 grid grid-cols-[auto_1fr_auto] items-center gap-3 text-xs">
      <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${dot}`} />
      <div className="min-w-0">
        <div className="truncate text-foreground/80">{meta.text}</div>
        {detail.length > 0 && (
          <div className="text-foreground/35 text-[10px] truncate">{detail.join(" · ")}</div>
        )}
      </div>
      <DualTime iso={event.ts} />
    </li>
  );
}

function PanelCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-foreground/8 bg-foreground/[0.015] overflow-hidden">
      {children}
    </div>
  );
}

function PanelHeader({
  title,
  count,
  subtitle,
}: {
  title: string;
  count?: number;
  subtitle?: string;
}) {
  return (
    <div className="px-5 py-2.5 text-[10px] uppercase tracking-widest font-mono text-foreground/50 border-b border-foreground/8 flex items-center justify-between gap-2">
      <span className="flex items-center gap-2">
        {title}
        {subtitle && <span className="text-foreground/25 normal-case tracking-normal">{subtitle}</span>}
      </span>
      {count !== undefined && (
        <span className="text-foreground/30 tabular-nums">{count}</span>
      )}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="px-5 py-8 text-center text-[10px] font-mono text-foreground/30">{text}</div>
  );
}

function BigStat({
  label,
  value,
  accent,
  span = 1,
}: {
  label: string;
  value: number | string;
  accent?: "emerald" | "orange" | "amber";
  span?: number;
}) {
  const valueClass =
    accent === "emerald" ? "text-emerald-300" :
    accent === "orange" ? "text-orange-300" :
    accent === "amber" ? "text-amber-300" :
    "text-foreground";
  return (
    <div
      className="rounded-2xl border border-foreground/8 bg-foreground/[0.015] px-4 py-3"
      style={span > 1 ? { gridColumn: `span ${span}` } : undefined}
    >
      <div className={["tabular-nums text-2xl font-light", valueClass].join(" ")}>{value}</div>
      <div className="mt-1 text-[10px] font-mono text-foreground/35 uppercase tracking-wider">{label}</div>
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-xl border border-foreground/8 bg-foreground/[0.015] px-4 py-3">
      <div className="tabular-nums text-xl font-light text-foreground">{value}</div>
      <div className="mt-0.5 text-[10px] font-mono text-foreground/35 uppercase tracking-wider">{label}</div>
    </div>
  );
}

function Sparkline({ buckets, height }: { buckets: number[]; height: number }) {
  const max = Math.max(...buckets, 1);
  const w = 100 / buckets.length;
  return (
    <svg
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
      aria-hidden
    >
      {buckets.map((v, i) => {
        const barH = (v / max) * height * 0.9;
        const x = i * w + w * 0.1;
        return (
          <rect
            key={i}
            x={x}
            y={height - barH}
            width={w * 0.8}
            height={barH}
            rx={1}
            className="fill-foreground/20"
          />
        );
      })}
    </svg>
  );
}

function JobRow({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const phase = job.progress.phase;
  const idx = job.progress.index;
  const total = job.progress.total ?? job.n_images;
  const img = job.progress.image;
  const subtitle =
    job.status === "running"
      ? `${phase ?? "running"}${idx ? ` ${idx}/${total ?? "?"}` : ""}${img ? ` · ${img}` : ""}`
      : job.status === "queued"
      ? `${total ?? job.n_images} img queued`
      : job.status === "failed"
      ? job.error ?? "failed"
      : job.status === "cancelled"
      ? "cancelled"
      : `${total ?? job.n_images} img`;
  const imageCount = total ?? job.n_images ?? 0;
  return (
    <li className="grid grid-cols-[5.5rem_5rem_3.5rem_1fr_7rem_5rem_5rem_5rem_4rem_2rem] gap-3 items-center px-5 py-2 text-xs">
      <StatusPill status={job.status} />
      <span className="text-foreground/40 uppercase tracking-widest text-[9px] font-mono">{KIND_LABEL[job.kind] ?? job.kind}</span>
      <span className="text-right tabular-nums text-foreground/70 font-mono">
        {imageCount > 0 ? imageCount.toLocaleString() : "—"}
      </span>
      <div className="min-w-0">
        <div className="truncate text-foreground/80 text-[11px]">{job.project}</div>
        <div className="truncate text-foreground/35 text-[10px] font-mono">{subtitle}</div>
      </div>
      <span className="truncate text-foreground/50 text-[11px] font-mono">@{job.user}</span>
      <span className="text-foreground/30 truncate font-mono text-[10px]">{job.id.slice(0, 8)}</span>
      <DualTime iso={job.queuedAt} />
      <span className="text-right tabular-nums text-foreground/70 font-mono">{formatElapsed(job.elapsedS)}</span>
      <span className="text-right tabular-nums text-foreground/70 font-mono">{formatCost(job.costPence)}</span>
      <span>
        {(job.status === "queued" || job.status === "running") && (
          <button
            onClick={onCancel}
            className="text-foreground/30 hover:text-red-300 transition-colors font-mono"
            title="Cancel"
          >
            ×
          </button>
        )}
      </span>
    </li>
  );
}

function StatusPill({ status }: { status: Job["status"] }) {
  const styles: Record<Job["status"], string> = {
    queued: "bg-amber-500/10 border border-amber-400/30 text-amber-300",
    running: "bg-orange-500/10 border border-orange-400/35 text-orange-300",
    done: "bg-foreground/5 border border-foreground/10 text-foreground/45",
    failed: "bg-red-500/10 border border-red-400/35 text-red-300",
    cancelled: "bg-foreground/5 border border-foreground/8 text-foreground/30",
    interrupted: "bg-amber-500/8 border border-amber-400/25 text-amber-300/70",
  };
  return (
    <span className={["rounded-full px-2 py-0.5 text-[9px] uppercase tracking-widest font-mono w-fit", styles[status]].join(" ")}>
      {status === "running" && <span className="inline-block h-1.5 w-1.5 mr-1 rounded-full bg-orange-300 animate-pulse" />}
      {status}
    </span>
  );
}

/** Shows "HH:MM" + "Xm ago" stacked, or single line on small columns. */
function DualTime({ iso }: { iso: string }) {
  const abs = absoluteTime(iso);
  const rel = relativeTime(iso);
  return (
    <div className="text-right shrink-0">
      <div className="text-[10px] font-mono text-foreground/50 tabular-nums">{abs}</div>
      <div className="text-[9px] font-mono text-foreground/30 tabular-nums">{rel}</div>
    </div>
  );
}

// ── Utility functions ─────────────────────────────────────────────────────────

function formatElapsed(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.floor(s - m * 60);
  return `${m}m${rem.toString().padStart(2, "0")}s`;
}

function relativeTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function absoluteTime(iso: string): string {
  const t = new Date(iso);
  if (!Number.isFinite(t.getTime())) return "—";
  const hh = t.getHours().toString().padStart(2, "0");
  const mm = t.getMinutes().toString().padStart(2, "0");
  const dd = t.getDate().toString().padStart(2, "0");
  const mo = (t.getMonth() + 1).toString().padStart(2, "0");
  return `${hh}:${mm} ${dd}/${mo}`;
}

function formatCost(p: number): string {
  return `$${(p / 100).toFixed(4)}`;
}

// ── Lock + Forbidden screens ──────────────────────────────────────────────────

function LockScreen({ onUnlock }: { onUnlock: (input: string) => Promise<string | null> }) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    if (busy || !value.trim()) return;
    setBusy(true);
    setErr(null);
    const failure = await onUnlock(value.trim());
    setBusy(false);
    if (failure) { setErr(failure); setValue(""); }
  };

  return (
    <main className="min-h-screen bg-[var(--background)] flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-3xl border border-foreground/10 bg-foreground/[0.02] overflow-hidden">
        <header className="px-6 pt-6 pb-3">
          <div className="text-[10px] text-orange-300 font-mono uppercase tracking-widest">Terminal</div>
          <h2 className="mt-1 text-2xl font-light tracking-tight">Restricted</h2>
          <p className="mt-2 text-sm text-foreground/45">
            Paste the access token printed at backend startup —{" "}
            <span className="font-mono text-foreground/70">[server] terminal token: …</span>
          </p>
        </header>
        <div className="px-6 pb-5">
          <input
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            type="password"
            placeholder="terminal token"
            spellCheck={false}
            autoComplete="off"
            className="w-full rounded-lg border border-foreground/15 bg-foreground/5 px-3 py-2.5 font-mono text-sm focus:outline-none focus:border-foreground/30 placeholder:text-foreground/25"
          />
          {err && <div className="mt-3 text-xs text-red-300 font-mono">{err}</div>}
        </div>
        <footer className="px-6 py-4 border-t border-foreground/8 flex justify-end">
          <button
            onClick={submit}
            disabled={busy || !value.trim()}
            className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Verifying…" : "Unlock"}
          </button>
        </footer>
      </div>
    </main>
  );
}

function Forbidden() {
  return (
    <main className="min-h-screen bg-[var(--background)] flex items-center justify-center p-6">
      <div className="max-w-md rounded-3xl border border-red-500/20 bg-red-500/[0.04] px-6 py-8 text-center">
        <div className="text-[10px] text-red-300 font-mono uppercase tracking-widest">Terminal</div>
        <h2 className="mt-1 text-2xl font-light tracking-tight">Access denied</h2>
        <p className="mt-2 text-sm text-foreground/45">
          Reserved for operator accounts:{" "}
          <span className="font-mono text-foreground/70">{ALLOWED_USERS.join(", ")}</span>
        </p>
      </div>
    </main>
  );
}
