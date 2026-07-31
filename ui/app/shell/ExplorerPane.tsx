"use client";

// Explorer side-bar pane: a per-workspace tree. Projects (containers)
// render as expandable nodes with their datasets as children; datasets
// that don't belong to any Project list at the root level.
//
// Data comes from the two endpoints the workspace already consumes:
//   GET /api/containers                → { containers: [{id, name, ...}] }
//   GET /api/projects?owner=<user>&…   → {total, items: ProjectSummary[]}
//     (each item: id, name, n_images, v2, createdBy,
//      container: {id, name} | null — the reverse map that assigns a
//      dataset to its Project)

import { useCallback, useEffect, useMemo, useState } from "react";

import { apiFetch } from "@/lib/apiFetch";
import { listContainers, type ContainerCard } from "@/lib/containers";

export type ExplorerDataset = {
  id: string;
  name: string;
  n_images: number;
  v2: boolean;
  owner: string;
  containerId: string | null;
};

// The slice of HomeView's ProjectSummary this pane reads.
type DatasetListItem = {
  id: string;
  name: string;
  n_images?: number;
  v2?: boolean;
  createdBy?: string;
  container?: { id: string; name: string } | null;
};

const STROKE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

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
  onOpenDataset,
  onNewDataset,
}: {
  username: string;
  selectedDatasetId: string | null;
  onOpenDataset: (ds: ExplorerDataset) => void;
  onNewDataset: () => void;
}) {
  const [containers, setContainers] = useState<ContainerCard[] | null>(null);
  const [datasets, setDatasets] = useState<ExplorerDataset[] | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
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

  const loading = datasets === null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Pane header: title + New dataset + Refresh. */}
      <div className="flex h-8 shrink-0 items-center justify-between pl-4 pr-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-foreground/55 select-none">
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
                  <button
                    type="button"
                    onClick={() => toggle(node.id)}
                    aria-expanded={open}
                    className="flex h-6 w-full items-center gap-1 px-2 text-left text-[13px] text-foreground/80 hover:bg-foreground/[0.05] transition-colors"
                  >
                    <Chevron open={open} />
                    <span className="min-w-0 flex-1 truncate">{node.name}</span>
                    <span className="pr-1 text-[11px] tabular-nums text-foreground/35">
                      {node.datasets.length}
                    </span>
                  </button>
                  {open &&
                    (node.datasets.length === 0 ? (
                      <p className="h-6 pl-8 pr-3 text-[12px] leading-6 text-foreground/35">
                        Empty
                      </p>
                    ) : (
                      node.datasets.map((ds) => (
                        <DatasetRow
                          key={ds.id}
                          ds={ds}
                          indent
                          selected={selectedDatasetId === ds.id}
                          onOpen={onOpenDataset}
                        />
                      ))
                    ))}
                </div>
              );
            })}
            {rootDatasets.map((ds) => (
              <DatasetRow
                key={ds.id}
                ds={ds}
                selected={selectedDatasetId === ds.id}
                onOpen={onOpenDataset}
              />
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function DatasetRow({
  ds,
  selected,
  indent = false,
  onOpen,
}: {
  ds: ExplorerDataset;
  selected: boolean;
  indent?: boolean;
  onOpen: (ds: ExplorerDataset) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onOpen(ds)}
      aria-current={selected || undefined}
      className={[
        "flex h-6 w-full items-center gap-2 pr-3 text-left text-[13px] transition-colors",
        indent ? "pl-8" : "pl-[1.375rem]",
        selected
          ? "bg-foreground/[0.08] text-[var(--foreground)]"
          : "text-foreground/75 hover:bg-foreground/[0.05] hover:text-foreground/95",
      ].join(" ")}
    >
      <span className="min-w-0 flex-1 truncate">{ds.name}</span>
      <span className="text-[11px] tabular-nums text-foreground/35">
        {ds.n_images}
      </span>
    </button>
  );
}
