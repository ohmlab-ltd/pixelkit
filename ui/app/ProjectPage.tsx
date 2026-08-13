"use client";

// One-page Project (container) view: a cinematic hero (cover/name/privacy)
// + a stat strip, the datasets in the project, models trained from them, and
// the activity timeline across all datasets. Opened from a Project card on
// the workspace. No user identity is rendered anywhere in this build.
// (Named ProjectPage, not ProjectView, since app/ProjectView.tsx is the
// legacy single-DATASET view.)
import { useCallback, useEffect, useState, type ReactNode } from "react";

import { apiFetch } from "@/lib/apiFetch";
import { ProjectSettingsModal } from "./ProjectSettingsModal";
import { AddDatasetModal } from "./AddDatasetModal";
import { CreateDatasetModal } from "./CreateDatasetModal";
import { GlassDialog } from "./v2/GlassDialog";
import { DerivedIcon } from "./v2/DerivedDatasets";
import {
  containerCoverUrl,
  datasetCoverUrl,
  deleteDataset,
  getActivity,
  getContainer,
  removeDataset,
  type ActivityItem,
  type ContainerDataset,
  type ContainerDetail,
} from "@/lib/containers";

function timeAgo(iso?: string): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function activityText(a: ActivityItem): string {
  const n = a.n_images ?? 0;
  switch (a.kind) {
    case "container_create":
      return "created the project";
    case "container_privacy":
      return "changed the project privacy";
    case "container_max_input":
      return "changed the image quality";
    case "member_add":
      return "added a member";
    case "member_remove":
      return "removed a member";
    case "dataset_add":
      return "added a dataset";
    case "dataset_remove":
      return "removed a dataset";
    case "job":
      if (a.job_kind === "upload") return `uploaded ${n} image${n === 1 ? "" : "s"}`;
      if (a.job_kind && a.job_kind.startsWith("label")) return `labelled ${n} image${n === 1 ? "" : "s"}`;
      if (a.job_kind === "train") return "trained a model";
      return a.job_kind ? `ran ${a.job_kind}` : "ran a job";
    default:
      return a.kind.replace(/_/g, " ");
  }
}

// ── tiny inline icons (stroke = currentColor) ────────────────────────────────
function IconGear() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

// Glassy pill used in the hero for metadata. Adapts to a dark-on-light variant
// when the cover's bottom band is light, so it stays legible.
function HeroPill({ children, light = false }: { children: ReactNode; light?: boolean }) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium backdrop-blur-md ring-1",
        light ? "bg-black/10 text-zinc-900 ring-black/10" : "bg-white/15 text-white ring-white/10",
      ].join(" ")}
    >
      {children}
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="pk-card rounded-2xl px-4 py-3.5">
      <div className="text-2xl font-bold tracking-tight tabular-nums text-[var(--foreground)]">
        {value.toLocaleString()}
      </div>
      <div className="pk-eyebrow mt-1">{label}</div>
    </div>
  );
}

function DatasetTile({ d, onClick, onRemove }: { d: ContainerDataset; onClick: () => void; onRemove?: () => void }) {
  const [coverFailed, setCoverFailed] = useState(false);
  return (
    <div className="pk-card pk-card-hover group relative flex flex-col overflow-hidden rounded-2xl">
      <button type="button" onClick={onClick} className="flex flex-col text-left">
        <div className="pk-cover relative aspect-[16/10] w-full overflow-hidden">
          {!coverFailed ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={datasetCoverUrl(d.id, d.updated)}
              alt=""
              loading="lazy"
              onError={() => setCoverFailed(true)}
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-3xl font-bold text-white/85 drop-shadow-sm transition-transform duration-300 group-hover:scale-110">
              {(d.name || "?").slice(0, 1).toUpperCase()}
            </div>
          )}
          {d.hasModel && (
            <span className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-full bg-emerald-500/90 px-2 py-0.5 text-[11px] font-semibold text-white shadow-sm">
              Model
            </span>
          )}
        </div>
        <div className="flex items-center justify-between gap-2 p-3.5">
          <span className="flex min-w-0 items-center gap-1.5 font-semibold text-foreground/90">
            <span className="truncate">{d.name}</span>
            {d.derived && (
              <span className="shrink-0 text-amber-600 dark:text-amber-400/80" title="Derived (cropped) dataset" aria-label="Derived dataset">
                <DerivedIcon className="h-3.5 w-3.5" />
              </span>
            )}
          </span>
          <span className="shrink-0 text-xs font-medium tabular-nums text-[var(--muted)]">
            {d.n_images.toLocaleString()}
          </span>
        </div>
      </button>
      {onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label="Remove from project"
          title="Remove from project (the dataset isn't deleted)"
          className="absolute right-2 top-2 grid h-7 w-7 place-items-center rounded-full bg-black/55 text-white opacity-0 backdrop-blur-md transition-opacity hover:bg-black/75 group-hover:opacity-100"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      )}
    </div>
  );
}

// Always try the cover; fall back to the monogram only on a real load error
// (the cover field has been arriving null even when a cover exists in R2). The
// `key` at the call site remounts this on cover change so it retries.
function HeroCover({ id, updated, monogram }: { id: string; updated?: string; monogram: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <span className="text-[7rem] font-bold leading-none text-white/25 drop-shadow">{monogram}</span>
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`${containerCoverUrl(id)}?v=${encodeURIComponent(updated || "")}`}
      alt=""
      onError={() => setFailed(true)}
      className="h-full w-full object-cover"
    />
  );
}

export function ProjectPage({
  containerId,
  username,
  onBack,
  onOpenDataset,
  onNewDataset,
  readOnly = false,
  backLabel = "Workspace",
}: {
  containerId: string;
  username: string;
  onBack: () => void;
  onOpenDataset?: (datasetId: string) => void;
  /** Start the full V2 onboarding for a new dataset in this project, with the
      name already entered here (the parent jumps to the labels stage + adds the
      result to the container). */
  onNewDataset?: (name: string) => void;
  /** Force a strictly read-only view (public guest landing): hides owner
      controls even if the viewer happens to be the owner. */
  readOnly?: boolean;
  backLabel?: string;
}) {
  const [detail, setDetail] = useState<ContainerDetail | null | "notfound">(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [showAddDataset, setShowAddDataset] = useState(false);
  const [showCreateDataset, setShowCreateDataset] = useState(false);
  const [copied, setCopied] = useState(false);
  // The dataset whose delete dialog (remove-from-project vs delete-entirely) is
  // open, plus an in-flight guard.
  const [deleteTarget, setDeleteTarget] = useState<ContainerDataset | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const reload = useCallback(() => {
    getContainer(containerId).then((d) => setDetail(d ?? "notfound"));
    getActivity(containerId).then(setActivity);
  }, [containerId]);
  useEffect(() => {
    reload();
  }, [reload]);

  if (detail === null) {
    return <div className="mx-auto max-w-6xl px-6 py-16 text-sm text-[var(--muted)]">Loading…</div>;
  }
  if (detail === "notfound") {
    return (
      <div className="mx-auto max-w-6xl px-6 py-16 text-center">
        <p className="text-[var(--muted)]">
          This project doesn&apos;t exist or you don&apos;t have access.
        </p>
        <button
          onClick={onBack}
          className="mt-4 rounded-xl bg-orange-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-orange-400"
        >
          Back to workspace
        </button>
      </div>
    );
  }

  const canManage = !readOnly && detail.my_role === "owner";
  // Editors (and owners) can contribute: add existing datasets + create new
  // ones in the Project. Only owners can MANAGE (rename/cover/members/delete).
  const canContribute = !readOnly && (detail.my_role === "owner" || detail.my_role === "editor");
  // Identity + role helpers for per-dataset delete permissions.
  const me = (username || "").trim().toLowerCase();
  const isOwnerRole = detail.my_role === "owner";
  const models = detail.datasets.filter((d) => d.hasModel);
  const totalImages = detail.datasets.reduce((s, d) => s + (d.n_images || 0), 0);
  const monogram = (detail.name || "?").slice(0, 1).toUpperCase();
  // A light cover gets a black title + light scrim; a dark cover (or no cover)
  // gets a white title + dark scrim, so the title always stands out.
  const lightCover = (detail.coverLuma ?? 0) > 145;

  return (
    <div className="pk-pop mx-auto max-w-6xl px-6 py-6">
      <button
        onClick={onBack}
        className="group mb-4 inline-flex items-center gap-1.5 text-sm font-medium text-[var(--muted)] transition-colors hover:text-foreground"
      >
        <span aria-hidden className="transition-transform group-hover:-translate-x-0.5">←</span> {backLabel}
      </button>

      {/* Hero */}
      <header className="relative mb-6 overflow-hidden rounded-3xl border border-foreground/10">
        <div className="pk-cover relative h-52 w-full sm:h-72">
          {/* Keyed/versioned on cover_updated (not detail.updated) so the hero
              only reloads when the cover image actually changes — editing the
              name / privacy / members no longer flickers the photo. */}
          <HeroCover key={detail.cover_updated ?? "cover"} id={detail.id} updated={detail.cover_updated} monogram={monogram} />
          <div
            className={`absolute inset-0 ${
              lightCover
                ? "bg-gradient-to-t from-white/85 via-white/35 to-transparent"
                : "bg-gradient-to-t from-black/75 via-black/25 to-black/5"
            }`}
          />

          {/* Top-right actions */}
          <div className="absolute right-4 top-4 flex items-center gap-2">
            {!readOnly && (
              <button
                type="button"
                onClick={() => {
                  const url = `${window.location.origin}/app?project=${detail.id}`;
                  navigator.clipboard
                    ?.writeText(url)
                    .then(() => {
                      setCopied(true);
                      setTimeout(() => setCopied(false), 1500);
                    })
                    .catch(() => {});
                }}
                className={`rounded-full px-3 py-1.5 text-xs font-medium ring-1 backdrop-blur-md transition ${
                  lightCover
                    ? "bg-black/10 text-zinc-900 ring-black/10 hover:bg-black/20"
                    : "bg-white/15 text-white ring-white/15 hover:bg-white/25"
                }`}
              >
                {copied ? "Copied ✓" : "Copy link"}
              </button>
            )}
            {canManage && (
              <button
                onClick={() => setShowSettings(true)}
                className="inline-flex items-center gap-1.5 rounded-full bg-white/90 px-3.5 py-1.5 text-sm font-semibold text-black shadow-sm backdrop-blur-md transition hover:bg-white"
              >
                <IconGear /> Settings
              </button>
            )}
          </div>

          {/* Bottom content: title -> owner -> metadata pills underneath. */}
          <div className="absolute inset-x-0 bottom-0 p-6">
            <h1 className={`pk-page-title text-4xl drop-shadow-sm sm:text-5xl ${lightCover ? "text-zinc-900" : "text-white"}`}>
              {detail.name}
            </h1>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <HeroPill light={lightCover}>
                {detail.datasets.length} dataset{detail.datasets.length === 1 ? "" : "s"}
              </HeroPill>
              {detail.updated && (
                <span className="hidden sm:inline-flex">
                  <HeroPill light={lightCover}>updated {timeAgo(detail.updated)}</HeroPill>
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Stat strip */}
      <div className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <StatCard label="Datasets" value={detail.datasets.length} />
        <StatCard label="Models" value={models.length} />
        <StatCard label="Images" value={totalImages} />
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        {/* Main: datasets + models */}
        <div className="lg:col-span-2">
          <section className="mb-9">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2.5 pk-section-title text-xl">
                <span className="pk-accent-bar" aria-hidden />
                Datasets
                <span className="rounded-full bg-foreground/[0.06] px-2 py-0.5 text-xs font-semibold text-[var(--muted)]">
                  {detail.datasets.length}
                </span>
              </h2>
              {canContribute && (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowAddDataset(true)}
                    className="rounded-lg border border-foreground/12 px-3 py-1.5 text-sm font-medium text-foreground/80 transition-colors hover:border-orange-400/50 hover:bg-orange-500/[0.05] hover:text-foreground"
                  >
                    Add existing
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowCreateDataset(true)}
                    className="rounded-lg bg-orange-500 px-3 py-1.5 text-sm font-semibold text-white transition hover:bg-orange-400"
                  >
                    + New dataset
                  </button>
                </div>
              )}
            </div>
            {detail.datasets.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-foreground/15 py-14 text-center text-sm text-[var(--muted)]">
                No datasets in this project yet.
                {canContribute && (
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <button
                      type="button"
                      onClick={() => setShowCreateDataset(true)}
                      className="rounded-xl bg-orange-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-orange-400"
                    >
                      Create a dataset
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowAddDataset(true)}
                      className="rounded-xl border border-foreground/12 px-4 py-2 text-sm font-medium text-foreground/80 hover:bg-foreground/5"
                    >
                      Add existing
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                {detail.datasets.map((d) => {
                  // Show the delete affordance when the viewer can do something
                  // with this dataset: the Project owner can detach any; a
                  // member can detach/delete only their own. The dialog then
                  // offers the specific allowed actions.
                  const iOwn = !!me && (d.owner || "").toLowerCase() === me;
                  const canRemove = !readOnly && (isOwnerRole || iOwn);
                  return (
                    <DatasetTile
                      key={d.id}
                      d={d}
                      onClick={() => onOpenDataset?.(d.id)}
                      onRemove={canRemove ? () => setDeleteTarget(d) : undefined}
                    />
                  );
                })}
              </div>
            )}
          </section>

          {models.length > 0 && (
            <section>
              <h2 className="mb-4 flex items-center gap-2.5 pk-section-title text-xl">
                <span className="pk-accent-bar" aria-hidden />
                Models
              </h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {models.map((d) => (
                  <div key={d.id} className="pk-card flex items-center gap-3 rounded-2xl p-4">
                    <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-emerald-500/12 text-emerald-600 dark:text-emerald-400">
                      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                        <path d="M12 2 4 6v6c0 5 3.5 8 8 10 4.5-2 8-5 8-10V6z" />
                        <path d="m9 12 2 2 4-4" />
                      </svg>
                    </span>
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-foreground/90">{d.name}</div>
                      <div className="text-xs text-[var(--muted)]">Trained model</div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* Side: labels + activity */}
        <aside className="flex flex-col gap-6">
          <LabelsCard containerId={containerId} />
          <section className="pk-card rounded-2xl p-5">
            <h2 className="pk-eyebrow mb-4">Activity</h2>
            {activity.length === 0 ? (
              <p className="text-sm text-[var(--muted)]">No activity yet.</p>
            ) : (
              <ol className="relative ml-1 flex flex-col gap-3.5 border-l border-foreground/10 pl-4">
                {activity.map((a, i) => (
                  <li key={i} className="relative text-sm leading-snug text-foreground/70">
                    <span
                      className="absolute -left-5 top-1 h-2 w-2 rounded-full bg-[var(--accent-orange)] ring-2 ring-[var(--surface)]"
                      aria-hidden
                    />
                    <span className="capitalize">{activityText(a)}</span>
                    <span className="ml-1 text-xs text-foreground/40">{timeAgo(a.ts)}</span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </aside>
      </div>

      {/* Delete a dataset: choose remove-from-project (detach) vs delete-entirely
          (destroy). Destroy is creator-only — even the Project owner can only
          detach someone else's dataset, never permanently delete it. */}
      <GlassDialog
        open={!!deleteTarget}
        onClose={() => { if (!deleteBusy) setDeleteTarget(null); }}
        title="Delete dataset"
        maxWidth="max-w-md"
      >
        {deleteTarget && (() => {
          const iOwn = !!me && (deleteTarget.owner || "").toLowerCase() === me;
          const canDetach = !readOnly && (isOwnerRole || iOwn);
          const canDestroy = !readOnly && iOwn;
          const finish = (p: Promise<boolean>) => {
            setDeleteBusy(true);
            void p.then(() => reload()).finally(() => { setDeleteBusy(false); setDeleteTarget(null); });
          };
          return (
            <div className="flex flex-col gap-3">
              <p className="-mt-1 text-sm text-[var(--muted)]">
                What should happen to <span className="font-semibold text-foreground/90">{deleteTarget.name}</span>?
              </p>
              {canDetach && (
                <button
                  type="button"
                  disabled={deleteBusy}
                  onClick={() => finish(removeDataset(detail.id, deleteTarget.id))}
                  className="flex flex-col gap-1 rounded-xl border border-foreground/12 p-3.5 text-left transition-colors hover:border-foreground/25 hover:bg-foreground/[0.03] disabled:opacity-50"
                >
                  <span className="text-sm font-semibold text-foreground/90">Remove from project</span>
                  <span className="text-xs text-[var(--muted)]">Takes it out of this project. The dataset and its images are kept as a standalone dataset.</span>
                </button>
              )}
              {canDestroy ? (
                <button
                  type="button"
                  disabled={deleteBusy}
                  onClick={() => finish(deleteDataset(deleteTarget.id))}
                  className="flex flex-col gap-1 rounded-xl border border-red-500/30 bg-red-500/[0.04] p-3.5 text-left transition-colors hover:border-red-500/50 hover:bg-red-500/[0.08] disabled:opacity-50"
                >
                  <span className="text-sm font-semibold text-red-600 dark:text-red-400">Delete entirely</span>
                  <span className="text-xs text-[var(--muted)]">Permanently deletes the dataset and all its images. This can&apos;t be undone.</span>
                </button>
              ) : (
                <p className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-3.5 py-2.5 text-xs text-[var(--muted)]">
                  Only the dataset&apos;s creator can permanently delete this dataset.
                </p>
              )}
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                disabled={deleteBusy}
                className="mt-1 self-end rounded-xl px-4 py-2 text-sm font-medium text-[var(--muted)] hover:bg-foreground/5 disabled:opacity-50"
              >
                Cancel
              </button>
            </div>
          );
        })()}
      </GlassDialog>

      {canManage && (
        <ProjectSettingsModal
          container={detail}
          open={showSettings}
          onClose={() => setShowSettings(false)}
          onChanged={() => reload()}
          onDeleted={() => {
            setShowSettings(false);
            onBack();
          }}
        />
      )}
      {canContribute && (
        <AddDatasetModal
          containerId={detail.id}
          username={username}
          existingIds={detail.datasets.map((d) => d.id)}
          open={showAddDataset}
          onClose={() => setShowAddDataset(false)}
          onAdded={() => reload()}
        />
      )}
      {canContribute && (
        <CreateDatasetModal
          open={showCreateDataset}
          onClose={() => setShowCreateDataset(false)}
          onContinue={(name) => {
            setShowCreateDataset(false);
            onNewDataset?.(name);
          }}
        />
      )}
    </div>
  );
}

// ── Labels (project-wide ontology) ───────────────────────────────────
// Usage of every label across the Project's datasets, with an inline
// rename that rewrites tags + annotations everywhere in one action.
// Renaming onto an existing label merges the two.

type LabelUsage = { label: string; datasets: number; boxes: number };

function LabelsCard({ containerId }: { containerId: string }) {
  const [rows, setRows] = useState<LabelUsage[] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await apiFetch(`/api/containers/${containerId}/labels`);
      if (r.ok) setRows(((await r.json()) as { labels: LabelUsage[] }).labels);
    } catch {
      /* engine unreachable — card shows nothing */
    }
  }, [containerId]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const commit = async (from: string) => {
    const to = draft.trim().toLowerCase();
    setEditing(null);
    if (!to || to === from) return;
    setBusy(true);
    setError(null);
    try {
      const r = await apiFetch(`/api/containers/${containerId}/labels/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_label: from, new_label: to }),
      });
      if (!r.ok) {
        const body = (await r.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? `rename failed (${r.status})`);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!rows || rows.length === 0) return null;
  const merging =
    editing !== null &&
    rows.some((r) => r.label === draft.trim().toLowerCase() && r.label !== editing);

  return (
    <section className="pk-card rounded-2xl p-5">
      <h2 className="pk-eyebrow mb-1.5">Labels</h2>
      <p className="mb-4 text-xs text-foreground/45">
        Across every dataset in this project. Rename to fix drift —
        renaming onto an existing label merges them.
      </p>
      {error && <p className="mb-2 text-xs text-[var(--bad)]">{error}</p>}
      <ul className="grid gap-1.5">
        {rows.map((r) => (
          <li
            key={r.label}
            className="flex items-center justify-between gap-2 rounded-md border border-[var(--line)] px-2.5 py-1.5"
          >
            {editing === r.label ? (
              <form
                className="flex min-w-0 flex-1 items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void commit(r.label);
                }}
              >
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={() => setEditing(null)}
                  onKeyDown={(e) => e.key === "Escape" && setEditing(null)}
                  className="w-full min-w-0 rounded border border-[var(--line-strong)] bg-transparent px-1.5 py-0.5 font-mono text-xs outline-none"
                />
                <span className="shrink-0 text-[10px] uppercase tracking-wider text-foreground/40">
                  {merging ? "merge ⏎" : "rename ⏎"}
                </span>
              </form>
            ) : (
              <>
                <span className="min-w-0 truncate font-mono text-xs text-foreground/85">
                  {r.label}
                </span>
                <span className="flex shrink-0 items-center gap-2.5 text-[11px] tabular-nums text-foreground/40">
                  {r.boxes} boxes · {r.datasets} ds
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => {
                      setEditing(r.label);
                      setDraft(r.label);
                    }}
                    className="text-[10px] uppercase tracking-wider text-foreground/50 transition-colors hover:text-foreground disabled:opacity-40"
                  >
                    Rename
                  </button>
                </span>
              </>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
