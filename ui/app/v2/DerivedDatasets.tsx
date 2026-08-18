"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "../../lib/apiFetch";
import { requestExplorerRefresh } from "../../lib/appNav";

// SPA navigation to another dataset: push the deep-link URL and fire a
// synthetic popstate - app/app/page.tsx's popstate handler swaps the
// open dataset view in place (same mechanism as the Explorer tree).
// No full page load.
function openDatasetSpa(projectId: string) {
  if (typeof window === "undefined") return;
  const target = `/app/${projectId}`;
  if (window.location.pathname !== target) {
    window.history.pushState(null, "", target);
  }
  window.dispatchEvent(new PopStateEvent("popstate"));
}

type Child = { project_id: string; name: string; labels: string[]; n_images: number };
type DerivedInfo = { parentProjectId?: string; parentName?: string };

// Suggest a child name from the PARENT dataset's name: title-case each word
// (preserving existing acronyms like "PPE") and append "Crops". So a dataset
// called "people" suggests "People Crops", "PPE" suggests "PPE Crops". Skips
// the suffix if the name already ends in "crop"/"crops".
function suggestCropName(parentName: string): string {
  const base = (parentName || "").trim();
  if (!base) return "";
  const titled = base
    .split(/\s+/)
    .map((w) => (w === w.toUpperCase() ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
  if (/crops?$/i.test(titled)) return titled;
  return `${titled} Crops`;
}

// A small "derived dataset" mark (distinct from the privacy padlock) for the
// title area + cards. A branch glyph: parent node forking to a child node.
export function DerivedIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={className} fill="none" aria-hidden>
      <circle cx="4" cy="3.2" r="1.7" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="4" cy="12.8" r="1.7" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="12" cy="12.8" r="1.7" stroke="currentColor" strokeWidth="1.3" />
      {/* Clean edge-to-edge connectors (no lines drawn through the circles).
          Vertical: top-left circle bottom (4,4.9) to bottom-left circle top
          (4,11.1). Horizontal: bottom-left circle right edge (5.7,12.8) to
          bottom-right circle left edge (10.3,12.8), on the bottom circles'
          centre line so it joins the middle of both. */}
      <path d="M4 4.9V11.1M5.7 12.8H10.3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

// Top-of-Dataset-tab bar. For a normal project: create a cropped child dataset
// + jump to existing children. For a DERIVED (child) project: a banner showing
// it's cropped from a parent, with a link to the parent + a manual re-sync.
export function DerivedDatasetsBar({ projectId, labels }: { projectId: string; labels: string[] }) {
  const [children, setChildren] = useState<Child[] | null>(null);
  const [derived, setDerived] = useState<DerivedInfo | null | undefined>(undefined); // undefined = loading
  const [ownName, setOwnName] = useState(""); // this project's own name (for the suggested child name)
  const [open, setOpen] = useState(false);
  const [syncing, setSyncing] = useState(false);

  // Is THIS project a derived child? (overview carries the link.) Also grab the
  // project's own name so the create-modal can pre-fill a smart child name.
  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/v2/projects/${projectId}/overview?imports_limit=1`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : {}))
      .then((d: { derived?: DerivedInfo | null; name?: string }) => {
        if (cancelled) return;
        setDerived(d?.derived || null);
        setOwnName(d?.name || "");
      })
      .catch(() => { if (!cancelled) setDerived(null); });
    return () => { cancelled = true; };
  }, [projectId]);

  // Only a parent lists children.
  useEffect(() => {
    if (derived) { setChildren(null); return; }
    let cancelled = false;
    apiFetch(`/api/v2/projects/${projectId}/children`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : { children: [] }))
      .then((d) => { if (!cancelled) setChildren(d.children || []); })
      .catch(() => { if (!cancelled) setChildren([]); });
    return () => { cancelled = true; };
  }, [projectId, derived]);

  const sync = async () => {
    setSyncing(true);
    try {
      await apiFetch(`/api/v2/projects/${projectId}/resync`, { method: "POST" });
      window.location.reload();
    } catch { setSyncing(false); }
  };

  if (derived === undefined) return null; // brief; avoids a flash of the wrong bar

  // ── derived (child) project banner ──
  if (derived) {
    return (
      <section className="mt-3 px-6 lg:px-10">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-3">
          <span className="pk-micro inline-flex items-center gap-1.5">
            <DerivedIcon className="h-3.5 w-3.5" /> Derived dataset
          </span>
          <span className="text-[12px] text-foreground/60">
            Cropped from <span className="font-medium text-foreground/80">{derived.parentName || "a parent project"}</span> · one label per image · auto-synced
          </span>
          <div className="ml-auto flex items-center gap-2">
            {derived.parentProjectId && (
              <a href={`/app/${derived.parentProjectId}`}
                onClick={(e) => { e.preventDefault(); if (derived.parentProjectId) openDatasetSpa(derived.parentProjectId); }}
                className="inline-flex h-8 items-center gap-1.5 rounded-md border border-[var(--line)] bg-transparent px-3 text-xs font-medium text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]">
                Open parent →
              </a>
            )}
            <button type="button" onClick={sync} disabled={syncing}
              className="h-8 rounded-md border border-[var(--line)] bg-transparent px-3 text-xs font-medium text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] disabled:opacity-50">
              {syncing ? "Syncing…" : "Sync now"}
            </button>
          </div>
        </div>
      </section>
    );
  }

  // ── normal/parent project: create + list children ──
  // Tertiary section on the Overview: a light heading + either compact rows
  // (when children exist) or a clear empty state, plus the create action.
  const count = children?.length ?? 0;
  const loading = children === null;
  return (
    <section className="mt-5 px-6 lg:px-10">
      <div className="pk-card rounded-md p-5">
        <div className="mb-3 flex items-end justify-between gap-3">
          <div className="min-w-0">
            <h2 className="pk-eyebrow">Derived datasets</h2>
            <p className="mt-1 text-[12px] text-foreground/60">
              Cropped, one-label-per-image children of this project.
            </p>
          </div>
          {/* A project can have MANY derived datasets (different label
              selections / crop settings), so the create button stays available
              even once children exist. Hidden only while the list is loading.
              (Deriving from a derived dataset is blocked server-side, and a
              child shows the banner above rather than this create UI.) */}
          {!loading && (
            <button type="button" onClick={() => setOpen(true)}
              className="h-9 shrink-0 rounded-md bg-[var(--accent)] px-4 text-[13px] font-medium text-[var(--accent-contrast)] transition-colors hover:brightness-105">
              + Create cropped dataset
            </button>
          )}
        </div>
        {loading ? (
          <div className="flex items-center justify-center gap-2 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-5 text-[13px] text-foreground/55">
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-foreground/20 border-t-[var(--accent)]" />
            Loading derived datasets…
          </div>
        ) : count === 0 ? (
          <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-6 text-center">
            <p className="text-[13px] font-medium text-foreground/70">No derived datasets yet</p>
            <p className="mt-1 text-[12px] text-foreground/55">
              Create a cropped child dataset to train on individual detections.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {children!.map((c) => (
              <a key={c.project_id} href={`/app/${c.project_id}`}
                onClick={(e) => { e.preventDefault(); openDatasetSpa(c.project_id); }}
                className="group flex items-center gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-3 transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]">
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-foreground/[0.06] text-foreground/70">
                  <DerivedIcon className="h-4 w-4" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-medium text-foreground">{c.name}</span>
                  <span className="block truncate text-[12px] text-foreground/60 tabular-nums">
                    {c.labels.length} label{c.labels.length === 1 ? "" : "s"} · {c.n_images} image{c.n_images === 1 ? "" : "s"}
                  </span>
                </span>
                <span className="shrink-0 text-foreground/40 transition-transform group-hover:translate-x-0.5" aria-hidden>→</span>
              </a>
            ))}
          </div>
        )}
      </div>
      {open && <DeriveModal projectId={projectId} labels={labels} parentName={ownName} onClose={() => setOpen(false)} />}
    </section>
  );
}

function DeriveModal({ projectId, labels, parentName, onClose }: { projectId: string; labels: string[]; parentName: string; onClose: () => void }) {
  // Pre-fill a smart suggestion from the parent name (e.g. "People Crops"); the
  // user can still overwrite it freely.
  const [name, setName] = useState(() => suggestCropName(parentName));
  const [sel, setSel] = useState<string[]>(labels);
  const [padding, setPadding] = useState(15);
  const [minSize, setMinSize] = useState(256);
  // ROI mode: force every crop to a 1:1 square (centred on the detection) so
  // long/thin objects become square ROIs instead of slivers.
  const [square, setSquare] = useState(false);
  // Optional fixed size: when on, every crop is resized to exactly the same
  // N x N size (forces square cropping). Off = crops keep their natural size.
  const [fixedSizeOn, setFixedSizeOn] = useState(false);
  const [fixedSizeVal, setFixedSizeVal] = useState(256);
  // Label source: "inherit" keeps the parent labels; "new" starts a fresh
  // vocabulary (crops come in unlabelled, parent label shown as a reference).
  const [labelMode, setLabelMode] = useState<"inherit" | "new">("inherit");
  // Whether to group the crop dataset (and its parent) under a workspace Project.
  // On = keep them together in a Project; off = create a standalone dataset.
  const [createProject, setCreateProject] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const toggle = (l: string) => setSel((s) => (s.includes(l) ? s.filter((x) => x !== l) : [...s, l]));

  const create = async () => {
    if (!name.trim()) { setErr("Give the dataset a name."); return; }
    if (sel.length === 0) { setErr("Pick at least one label."); return; }
    setBusy(true); setErr(null);
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("labels", JSON.stringify(sel));
      fd.append("padding", String(padding / 100));
      fd.append("min_size", String(minSize));
      fd.append("square", square ? "true" : "false");
      fd.append("fixed_size", fixedSizeOn ? String(fixedSizeVal) : "0");
      fd.append("label_mode", labelMode);
      fd.append("create_project", createProject ? "true" : "false");
      const r = await apiFetch(`/api/v2/projects/${projectId}/derive`, { method: "POST", body: fd });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `http ${r.status}`);
      }
      const d = await r.json() as { project_id: string };
      // New child (and possibly a new wrapping Project) exists - refresh
      // the Explorer tree, close the dialog, and open the child in-app.
      requestExplorerRefresh();
      onClose();
      openDatasetSpa(d.project_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  if (typeof document === "undefined") return null;
  // Portal to <body> so a transformed ancestor (e.g. the tab's pk-up entrance
  // animation) can't break `position: fixed` and push the modal off-screen.
  return createPortal(
    <div className="pk-backdrop fixed inset-0 z-[1300] flex items-start justify-center overflow-y-auto p-4 pt-[8vh]" onClick={onClose}>
      <div className="pk-glass pk-pop w-full max-w-md rounded-md p-5" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-[15px] font-medium tracking-tight">Create cropped dataset</h3>
        <p className="mt-1 text-[13px] leading-relaxed text-foreground/55">
          A new project of <span className="font-medium text-foreground/75">one image per detection</span>, each cropped to its
          box with a single label. It auto-syncs from this project (one-way: edits here flow down, never back up).
        </p>

        <label className="mt-4 block">
          <span className="text-[12px] font-medium text-foreground/70">Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. PPE crops"
            className="mt-1 w-full rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm outline-none focus:border-[var(--line-strong)]" />
        </label>

        <div className="mt-4">
          <span className="text-[12px] font-medium text-foreground/70">Label source</span>
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            {([
              { key: "inherit", title: "Keep parent labels", desc: "Crops carry the parent's label." },
              { key: "new", title: "Create new labels", desc: "Start fresh - crops come in unlabelled." },
            ] as const).map((opt) => {
              const on = labelMode === opt.key;
              return (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => setLabelMode(opt.key)}
                  className={`rounded-md border p-2.5 text-left transition-colors ${on ? "border-[var(--accent)] bg-[var(--accent-dim)]" : "border-[var(--line)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]"}`}
                >
                  <span className={`block text-[12.5px] font-medium ${on ? "text-[var(--foreground)]" : "text-foreground/85"}`}>{opt.title}</span>
                  <span className="mt-0.5 block text-[11px] leading-snug text-foreground/50">{opt.desc}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-4">
          <span className="text-[12px] font-medium text-foreground/70">
            {labelMode === "new" ? "Detections to crop" : "Labels to include"}
          </span>
          <div className="mt-1.5 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
            {labels.length === 0 && <span className="text-[12px] text-foreground/40">This project has no labels yet.</span>}
            {labels.map((l) => {
              const on = sel.includes(l);
              return (
                <button key={l} type="button" onClick={() => toggle(l)}
                  className={`inline-flex h-6 items-center rounded-md border px-2 font-mono text-[12px] transition-colors ${on ? "border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--foreground)]" : "border-[var(--line)] bg-transparent text-[var(--fg-soft)] hover:border-[var(--line-strong)]"}`}>
                  {l}
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="min-w-0">
            <span className="block text-[12px] font-medium text-foreground/70">ROI mode - square crops</span>
            <span className="mt-0.5 block text-[11px] leading-snug text-foreground/40">Force every crop to an exact 1:1 square (centred on the object) so long, thin objects aren&apos;t cropped to slivers.</span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={square}
            onClick={() => setSquare((s) => !s)}
            className={`relative flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors ${square ? "justify-end bg-[var(--accent)]" : "justify-start bg-foreground/20"}`}
            title={square ? "Square (1:1) crops" : "Crop tight to each box"}
          >
            <span className="h-4 w-4 rounded-full bg-white shadow-sm" />
          </button>
        </div>

        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="min-w-0">
            <span className="block text-[12px] font-medium text-foreground/70">Group in a workspace Project</span>
            <span className="mt-0.5 block text-[11px] leading-snug text-foreground/40">Keeps the crop dataset alongside its parent in a Project. Off creates a standalone dataset.</span>
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={createProject}
            onClick={() => setCreateProject((s) => !s)}
            className={`relative flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors ${createProject ? "justify-end bg-[var(--accent)]" : "justify-start bg-foreground/20"}`}
            title={createProject ? "Grouped in a Project" : "Standalone dataset"}
          >
            <span className="h-4 w-4 rounded-full bg-white shadow-sm" />
          </button>
        </div>

        <label className="mt-4 block">
          <span className="flex items-center justify-between text-[12px] font-medium text-foreground/70">
            <span>Context padding</span><span className="tabular-nums text-foreground/50">{padding}%</span>
          </span>
          <input type="range" min={0} max={50} step={5} value={padding} onChange={(e) => setPadding(Number(e.target.value))}
            className="mt-1.5 w-full accent-[var(--accent)]" />
          <span className="text-[11px] text-foreground/40">0% = tight to the box; higher keeps surrounding context.</span>
        </label>

        <label className="mt-4 block">
          <span className="flex items-center justify-between text-[12px] font-medium text-foreground/70">
            <span>Minimum image size</span><span className="tabular-nums text-foreground/50">{minSize}px</span>
          </span>
          <input type="range" min={64} max={512} step={32} value={minSize} onChange={(e) => setMinSize(Number(e.target.value))}
            className="mt-1.5 w-full accent-[var(--accent)]" />
          <span className="text-[11px] text-foreground/40">Crops smaller than this are scaled up so every image is at least {minSize}x{minSize}.</span>
        </label>

        <div className="mt-4">
          <div className="flex items-center justify-between gap-3">
            <span className="min-w-0">
              <span className="block text-[12px] font-medium text-foreground/70">Fixed crop size</span>
              <span className="mt-0.5 block text-[11px] leading-snug text-foreground/40">Resize every crop to one fixed square size, so all images match. Forces square crops and overrides the minimum size.</span>
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={fixedSizeOn}
              onClick={() => setFixedSizeOn((s) => !s)}
              className={`relative flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors ${fixedSizeOn ? "justify-end bg-[var(--accent)]" : "justify-start bg-foreground/20"}`}
              title={fixedSizeOn ? "All crops the same fixed size" : "Crops keep their natural per-detection size"}
            >
              <span className="h-4 w-4 rounded-full bg-white shadow-sm" />
            </button>
          </div>
          {fixedSizeOn && (
            <label className="mt-3 block">
              <span className="flex items-center justify-between text-[12px] font-medium text-foreground/70">
                <span>Size</span><span className="tabular-nums text-foreground/50">{fixedSizeVal}px</span>
              </span>
              <input type="range" min={64} max={512} step={32} value={fixedSizeVal} onChange={(e) => setFixedSizeVal(Number(e.target.value))}
                className="mt-1.5 w-full accent-[var(--accent)]" />
              <span className="text-[11px] text-foreground/40">Every crop becomes exactly {fixedSizeVal}x{fixedSizeVal}.</span>
            </label>
          )}
        </div>

        {err && <div className="mt-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[13px] text-[var(--bad)]">{err}</div>}

        <div className="mt-5 flex justify-end gap-2">
          <button type="button" onClick={onClose} className="h-9 rounded-md border border-[var(--line)] px-4 text-sm font-medium text-[var(--fg-soft)] transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)] hover:text-foreground">Cancel</button>
          <button type="button" disabled={busy} onClick={create} className="h-9 rounded-md bg-[var(--accent)] px-4 text-sm font-medium text-[var(--accent-contrast)] transition-colors hover:brightness-105 disabled:opacity-40">
            {busy ? "Creating…" : "Create"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
