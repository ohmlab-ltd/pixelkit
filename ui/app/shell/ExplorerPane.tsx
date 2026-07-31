"use client";

// Explorer side-bar pane: THE single navigation tree for the app.
// Three levels:
//   Projects (containers)      → expandable, children are datasets
//     Datasets                 → V2 datasets expand one level further
//       Sections               → Overview / References / Dataset /
//                                Augmentations (the old in-view
//                                SidebarNav, folded into the tree)
// Datasets that don't belong to any Project list at the root level.
//
// Data comes from the two endpoints the workspace already consumes:
//   GET /api/containers                → { containers: [{id, name, ...}] }
//   GET /api/projects?owner=<user>&…   → {total, items: ProjectSummary[]}
//     (each item: id, name, n_images, n_references, derived, v2,
//      createdBy, container: {id, name} | null — the reverse map that
//      assigns a dataset to its Project)

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiFetch } from "@/lib/apiFetch";
import { onExplorerRefresh } from "@/lib/appNav";
import { listContainers, type ContainerCard } from "@/lib/containers";

export type ExplorerDataset = {
  id: string;
  name: string;
  n_images: number;
  n_references: number;
  derived: boolean;
  v2: boolean;
  owner: string;
  containerId: string | null;
};

// Dataset view sections, third tree level. Mirrors the V2 dataset
// view's ProjectTab union (the view consumes these via its `section`
// prop; app/page.tsx owns the state).
export type DatasetSection = "overview" | "references" | "dataset" | "augmentations";

// The slice of HomeView's ProjectSummary this pane reads.
type DatasetListItem = {
  id: string;
  name: string;
  n_images?: number;
  n_references?: number;
  derived?: { parentProjectId?: string; parentName?: string } | null;
  v2?: boolean;
  createdBy?: string;
  container?: { id: string; name: string } | null;
};

// Section rows under an expanded V2 dataset. Availability mirrors the
// old in-view SidebarNav rules for an editable dataset: References is
// disabled on derived datasets (they don't manage their own reference
// images) and carries the reference count; Dataset carries the image
// count. (The portable build is single-user, so the read-only variant
// of those rules never applies here.)
function sectionsFor(ds: ExplorerDataset): {
  key: DatasetSection;
  label: string;
  count: number | null;
  disabled: boolean;
  disabledHint?: string;
}[] {
  return [
    { key: "overview", label: "Overview", count: null, disabled: false },
    {
      key: "references",
      label: "References",
      count: ds.n_references,
      disabled: ds.derived,
      disabledHint: "Derived datasets don't manage their own reference images",
    },
    { key: "dataset", label: "Dataset", count: ds.n_images, disabled: false },
    { key: "augmentations", label: "Augmentations", count: null, disabled: false },
  ];
}

const STROKE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

// Per-section line icons for the third-level tree rows. Shapes come
// from the old in-view SidebarNav (grid / image / layers / sparkle),
// compacted to 14px and muted via opacity so they track each row's
// own text colour (idle / active / disabled) while staying quieter
// than the 13px label beside them.
function SectionIcon({ section }: { section: DatasetSection }) {
  const common = {
    viewBox: "0 0 24 24",
    width: 14,
    height: 14,
    ...STROKE,
    className: "shrink-0 opacity-60",
    "aria-hidden": true,
  };
  switch (section) {
    case "overview":
      return (
        <svg {...common}>
          <rect x="3" y="3" width="7" height="7" rx="1.5" />
          <rect x="14" y="3" width="7" height="7" rx="1.5" />
          <rect x="3" y="14" width="7" height="7" rx="1.5" />
          <rect x="14" y="14" width="7" height="7" rx="1.5" />
        </svg>
      );
    case "references":
      return (
        <svg {...common}>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M3 16l5-5 4 4 3-3 6 6" />
          <circle cx="9" cy="9" r="1.4" />
        </svg>
      );
    case "dataset":
      return (
        <svg {...common}>
          <path d="M4 7l8-4 8 4-8 4-8-4z" />
          <path d="M4 12l8 4 8-4" />
          <path d="M4 17l8 4 8-4" />
        </svg>
      );
    case "augmentations":
      return (
        <svg {...common}>
          <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18" />
          <circle cx="12" cy="12" r="2.5" />
        </svg>
      );
  }
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={14}
      height={14}
      {...STROKE}
      aria-hidden
      className={["shrink-0 transition-transform", open ? "rotate-90" : ""].join(" ")}
    >
      <path d="m9 6 6 6-6 6" />
    </svg>
  );
}

export function ExplorerPane({
  username,
  selectedDatasetId,
  activeSection,
  onOpenDataset,
  onOpenSection,
  onOpenProject,
  onNewDataset,
  onCollapse,
}: {
  username: string;
  selectedDatasetId: string | null;
  activeSection: DatasetSection | null;
  onOpenDataset: (ds: ExplorerDataset) => void;
  onOpenSection: (ds: ExplorerDataset, section: DatasetSection) => void;
  /** Open the Project (container) page for a top-level tree row.
   *  Fired by the row's NAME area only — the chevron just expands. */
  onOpenProject: (containerId: string) => void;
  onNewDataset: () => void;
  /** Collapse the side bar (pane-header chevron button). The
   *  activity bar's Explorer icon re-expands it. */
  onCollapse: () => void;
}) {
  const [containers, setContainers] = useState<ContainerCard[] | null>(null);
  const [datasets, setDatasets] = useState<ExplorerDataset[] | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // Expanded DATASET nodes (third tree level). Kept separately from the
  // project set so a project and a dataset sharing an id prefix can
  // never collide, and so "collapse the open dataset" stays possible.
  const [dsExpanded, setDsExpanded] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [cards, dsRes] = await Promise.all([
        listContainers(),
        apiFetch(
          `/api/projects?owner=${encodeURIComponent(username)}&viewer=${encodeURIComponent(username)}&offset=0&limit=1000`,
          { cache: "no-store" },
        ),
      ]);
      if (!dsRes.ok) throw new Error(`http ${dsRes.status}`);
      const payload = (await dsRes.json()) as
        | DatasetListItem[]
        | { total: number; items: DatasetListItem[] };
      const items: DatasetListItem[] = Array.isArray(payload)
        ? payload
        : (payload.items ?? []);
      setDatasets(
        items.map((p) => ({
          id: p.id,
          name: p.name,
          n_images: p.n_images ?? 0,
          n_references: p.n_references ?? 0,
          derived: !!p.derived,
          v2: !!p.v2,
          owner: p.createdBy ?? "",
          containerId: p.container?.id ?? null,
        })),
      );
      // listContainers() returns null on failure — keep the last-known
      // list rather than clobbering the tree with an empty state.
      if (cards !== null) setContainers(cards);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [username]);

  useEffect(() => {
    refresh();
    // Light poll to keep names/counts in sync with the workspace's own
    // 4s poll; the backend caches this response so the cost is small.
    const id = window.setInterval(refresh, 10000);
    return () => window.clearInterval(id);
  }, [refresh]);

  // Immediate refresh on the appNav bus signal. Mutation paths
  // (dataset create / delete / duplicate / rename, add-to-Project)
  // fire it the moment their API call resolves, so the tree reflects
  // the change right away instead of waiting out the 10 s poll.
  useEffect(() => onExplorerRefresh(() => { void refresh(); }), [refresh]);

  // Opening a dataset (from anywhere — tree, workspace cards, deep
  // link) reveals its section rows: expand the dataset node and the
  // Project that contains it, so the selection is always visible.
  // Guarded by a "handled" ref so the 10s poll's fresh `datasets`
  // array can't keep re-expanding a node the user chose to collapse;
  // only an ACTUAL selection change re-expands. The container lookup
  // needs the listing, so handling is deferred until it has loaded
  // (deep-link case: selection lands before the first fetch).
  const expandHandledRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedDatasetId) {
      expandHandledRef.current = null;
      return;
    }
    if (expandHandledRef.current === selectedDatasetId) return;
    setDsExpanded((prev) =>
      prev.has(selectedDatasetId) ? prev : new Set(prev).add(selectedDatasetId),
    );
    if (datasets === null) return; // container unknown — retry once loaded
    // A just-created dataset can be selected BEFORE the listing has
    // caught up (the create path fires an explorer-refresh, but this
    // effect may run against the stale array first). Don't mark it
    // handled until the dataset actually appears — the refreshed
    // `datasets` re-runs this effect and the container then expands.
    const ds = datasets.find((d) => d.id === selectedDatasetId);
    if (!ds) return;
    expandHandledRef.current = selectedDatasetId;
    if (ds.containerId) {
      const container = ds.containerId;
      setExpanded((prev) => (prev.has(container) ? prev : new Set(prev).add(container)));
    }
  }, [selectedDatasetId, datasets]);

  // Group datasets under their Project; anything unassigned lists at root.
  const { projectNodes, rootDatasets } = useMemo(() => {
    const byContainer = new Map<string, ExplorerDataset[]>();
    const root: ExplorerDataset[] = [];
    for (const d of datasets ?? []) {
      if (d.containerId) {
        const list = byContainer.get(d.containerId) ?? [];
        list.push(d);
        byContainer.set(d.containerId, list);
      } else {
        root.push(d);
      }
    }
    const nodes = (containers ?? []).map((c) => ({
      id: c.id,
      name: c.name,
      datasets: byContainer.get(c.id) ?? [],
    }));
    // Containers referenced by a dataset but missing from the listing
    // (e.g. race between the two fetches) still get a node so their
    // datasets don't vanish from the tree.
    const known = new Set(nodes.map((n) => n.id));
    for (const [cid, list] of byContainer) {
      if (!known.has(cid)) {
        nodes.push({ id: cid, name: "Project", datasets: list });
      }
    }
    return { projectNodes: nodes, rootDatasets: root };
  }, [containers, datasets]);

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // Expand-only variant for the project NAME rows: navigation is the
  // primary action there, so a click may reveal the children but must
  // never collapse them (collapsing is the chevron's job).
  const expand = (id: string) =>
    setExpanded((prev) => (prev.has(id) ? prev : new Set(prev).add(id)));

  const toggleDs = (id: string) =>
    setDsExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const loading = datasets === null;

  const renderDataset = (ds: ExplorerDataset, indent: boolean) => (
    <DatasetNode
      key={ds.id}
      ds={ds}
      indent={indent}
      selected={selectedDatasetId === ds.id}
      activeSection={selectedDatasetId === ds.id ? activeSection : null}
      expanded={dsExpanded.has(ds.id)}
      onToggle={() => toggleDs(ds.id)}
      onOpen={onOpenDataset}
      onOpenSection={onOpenSection}
    />
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Pane header: title + New dataset + Refresh + Collapse. */}
      <div className="flex h-8 shrink-0 items-center justify-between pl-4 pr-2">
        <span className="pk-micro select-none">
          Explorer
        </span>
        <span className="flex items-center">
          <button
            type="button"
            title="New dataset"
            aria-label="New dataset"
            onClick={onNewDataset}
            className="grid h-6 w-6 place-items-center rounded text-foreground/55 hover:bg-foreground/[0.08] hover:text-foreground/90 transition-colors"
          >
            <svg viewBox="0 0 24 24" width={14} height={14} {...STROKE} aria-hidden>
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
          <button
            type="button"
            title="Refresh"
            aria-label="Refresh explorer"
            onClick={() => void refresh()}
            className="grid h-6 w-6 place-items-center rounded text-foreground/55 hover:bg-foreground/[0.08] hover:text-foreground/90 transition-colors"
          >
            <svg viewBox="0 0 24 24" width={14} height={14} {...STROKE} aria-hidden>
              <path d="M21 12a9 9 0 1 1-2.64-6.36" />
              <path d="M21 3v6h-6" />
            </svg>
          </button>
          <button
            type="button"
            title="Collapse Explorer"
            aria-label="Collapse Explorer"
            onClick={onCollapse}
            className="grid h-6 w-6 place-items-center rounded text-foreground/55 hover:bg-foreground/[0.08] hover:text-foreground/90 transition-colors"
          >
            <svg viewBox="0 0 24 24" width={14} height={14} {...STROKE} aria-hidden>
              <path d="m15 6-6 6 6 6" />
            </svg>
          </button>
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        {loading ? (
          <p className="px-4 py-1 text-[13px] text-foreground/40">Loading…</p>
        ) : error && projectNodes.length === 0 && rootDatasets.length === 0 ? (
          <p className="px-4 py-1 text-[13px] text-foreground/40">
            Engine unreachable.
          </p>
        ) : projectNodes.length === 0 && rootDatasets.length === 0 ? (
          <p className="px-4 py-1 text-[13px] text-foreground/40">
            No datasets yet.
          </p>
        ) : (
          <>
            {projectNodes.map((node) => {
              const open = expanded.has(node.id);
              return (
                <div key={node.id}>
                  {/* Two sibling buttons, same pattern as DatasetNode:
                      the chevron ONLY toggles expansion, the name row
                      opens the Project page (and reveals the children
                      as a side effect — expand, never collapse). */}
                  <div className="flex w-full items-stretch pl-2">
                    <button
                      type="button"
                      onClick={() => toggle(node.id)}
                      aria-expanded={open}
                      aria-label={`${open ? "Collapse" : "Expand"} ${node.name}`}
                      className="grid w-[18px] shrink-0 place-items-center text-foreground/55 hover:text-foreground/90 transition-colors"
                    >
                      <Chevron open={open} />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        expand(node.id);
                        onOpenProject(node.id);
                      }}
                      className="flex h-6 min-w-0 flex-1 items-center gap-1 pr-2 text-left text-[13px] text-foreground/80 hover:bg-foreground/[0.05] transition-colors"
                    >
                      <span className="min-w-0 flex-1 truncate">{node.name}</span>
                      <span className="pr-1 font-mono text-[11px] tabular-nums text-foreground/35">
                        {node.datasets.length}
                      </span>
                    </button>
                  </div>
                  {open &&
                    (node.datasets.length === 0 ? (
                      <p className="h-6 pl-8 pr-3 text-[12px] leading-6 text-foreground/35">
                        Empty
                      </p>
                    ) : (
                      node.datasets.map((ds) => renderDataset(ds, true))
                    ))}
                </div>
              );
            })}
            {rootDatasets.map((ds) => renderDataset(ds, false))}
          </>
        )}
      </div>
    </div>
  );
}

// One dataset in the tree plus (for V2 datasets) its expandable
// section rows. V1 datasets have no sections — the legacy view keeps
// its own internal nav — so they render as a plain row.
function DatasetNode({
  ds,
  selected,
  activeSection,
  expanded,
  indent,
  onToggle,
  onOpen,
  onOpenSection,
}: {
  ds: ExplorerDataset;
  selected: boolean;
  activeSection: DatasetSection | null;
  expanded: boolean;
  indent: boolean;
  onToggle: () => void;
  onOpen: (ds: ExplorerDataset) => void;
  onOpenSection: (ds: ExplorerDataset, section: DatasetSection) => void;
}) {
  const hasSections = ds.v2;
  const showSections = hasSections && expanded;
  // While the section rows are visible, the ACTIVE SECTION row carries
  // the selected highlight; the dataset row itself only keeps the
  // full-strength text so the highlight isn't doubled.
  const rowHighlight = selected && !showSections;
  const nameCls = [
    "flex h-6 min-w-0 flex-1 items-center gap-2 pr-3 text-left text-[13px] transition-colors",
    rowHighlight
      ? "bg-foreground/[0.08] text-[var(--foreground)]"
      : selected
        ? "text-[var(--foreground)] hover:bg-foreground/[0.05]"
        : "text-foreground/75 hover:bg-foreground/[0.05] hover:text-foreground/95",
  ].join(" ");

  return (
    <>
      {hasSections ? (
        // Two sibling buttons (chevron + name) — nesting a button in a
        // button is invalid HTML. Chevron toggles the section rows,
        // the name row opens the dataset (which also expands it via
        // the selection effect in the pane).
        <div className={["flex w-full items-stretch", indent ? "pl-3.5" : "pl-2"].join(" ")}>
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={expanded}
            aria-label={`${expanded ? "Collapse" : "Expand"} ${ds.name}`}
            className="grid w-[18px] shrink-0 place-items-center text-foreground/55 hover:text-foreground/90 transition-colors"
          >
            <Chevron open={expanded} />
          </button>
          <button
            type="button"
            onClick={() => onOpen(ds)}
            aria-current={selected || undefined}
            className={nameCls}
          >
            <span className="min-w-0 flex-1 truncate">{ds.name}</span>
            <span className="font-mono text-[11px] tabular-nums text-foreground/35">
              {ds.n_images}
            </span>
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => onOpen(ds)}
          aria-current={selected || undefined}
          className={[nameCls, "w-full", indent ? "pl-8" : "pl-[1.625rem]"].join(" ")}
        >
          <span className="min-w-0 flex-1 truncate">{ds.name}</span>
          <span className="font-mono text-[11px] tabular-nums text-foreground/35">
            {ds.n_images}
          </span>
        </button>
      )}
      {showSections &&
        sectionsFor(ds).map((s) => {
          const active = selected && activeSection === s.key;
          return (
            <button
              key={s.key}
              type="button"
              disabled={s.disabled}
              title={s.disabled ? s.disabledHint : undefined}
              onClick={() => { if (!s.disabled) onOpenSection(ds, s.key); }}
              aria-current={active ? "true" : undefined}
              className={[
                "flex h-6 w-full items-center gap-2 pr-3 text-left text-[13px] transition-colors",
                indent ? "pl-[3.125rem]" : "pl-11",
                s.disabled
                  ? "cursor-not-allowed text-foreground/25"
                  : active
                    ? "bg-foreground/[0.08] text-[var(--foreground)]"
                    : "text-foreground/70 hover:bg-foreground/[0.05] hover:text-foreground/95",
              ].join(" ")}
            >
              <SectionIcon section={s.key} />
              <span className="min-w-0 flex-1 truncate">{s.label}</span>
              {typeof s.count === "number" && !s.disabled && (
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-foreground/35">
                  {s.count}
                </span>
              )}
            </button>
          );
        })}
    </>
  );
}
