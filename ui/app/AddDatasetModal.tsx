"use client";

// Pick datasets you own and move them into a Project (container). Datasets
// inherit the Project's privacy on the way in. Owner-only (the backend requires
// container-manage + dataset ownership).
import { useEffect, useRef, useState } from "react";

import { GlassDialog } from "./v2/GlassDialog";
import { apiFetch } from "@/lib/apiFetch";
import { addDataset } from "@/lib/containers";

type Ds = { id: string; name: string };

export function AddDatasetModal({
  containerId,
  username,
  existingIds,
  open,
  onClose,
  onAdded,
}: {
  containerId: string;
  username: string;
  existingIds: string[];
  open: boolean;
  onClose: () => void;
  onAdded: () => void;
}) {
  const [datasets, setDatasets] = useState<Ds[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  // Ref so the fetch effect depends only on [open, username] and doesn't re-run
  // every render (existingIds is a fresh array each time).
  const existingRef = useRef(existingIds);
  existingRef.current = existingIds;

  useEffect(() => {
    if (!open || !username) return;
    setSelected(new Set());
    setDatasets(null);
    apiFetch(
      `/api/projects?owner=${encodeURIComponent(username)}&viewer=${encodeURIComponent(username)}&offset=0&limit=300`,
    )
      .then((r) => (r.ok ? r.json() : null))
      .then((payload: unknown) => {
        const arr: Array<{ id?: string; name?: string }> = Array.isArray(payload)
          ? (payload as Array<{ id?: string; name?: string }>)
          : ((payload as { items?: Array<{ id?: string; name?: string }> } | null)?.items ?? []);
        const existing = new Set(existingRef.current);
        setDatasets(
          arr
            .filter((p) => p && p.id && !existing.has(p.id))
            .map((p) => ({ id: p.id as string, name: p.name || (p.id as string) })),
        );
      })
      .catch(() => setDatasets([]));
  }, [open, username]);

  function toggle(id: string) {
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }

  async function submit() {
    if (selected.size === 0) return;
    setBusy(true);
    for (const id of Array.from(selected)) {
      await addDataset(containerId, id);
    }
    setBusy(false);
    onAdded();
    onClose();
  }

  return (
    <GlassDialog open={open} onClose={onClose} title="Add datasets to project" maxWidth="max-w-md">
      <div className="flex flex-col gap-4">
        <p className="-mt-1 text-sm text-[var(--muted)]">
          Pick datasets you own to move into this project. They inherit the project&apos;s privacy.
        </p>
        <div className="max-h-72 overflow-y-auto rounded-xl border border-foreground/10">
          {datasets === null ? (
            <div className="p-4 text-sm text-foreground/40">Loading…</div>
          ) : datasets.length === 0 ? (
            <div className="p-4 text-sm text-foreground/40">No datasets available to add.</div>
          ) : (
            <ul className="divide-y divide-foreground/5">
              {datasets.map((d) => (
                <li key={d.id}>
                  <label className="flex cursor-pointer items-center gap-3 px-3 py-2.5 hover:bg-foreground/[0.04]">
                    <input
                      type="checkbox"
                      checked={selected.has(d.id)}
                      onChange={() => toggle(d.id)}
                      className="h-4 w-4 accent-orange-500"
                    />
                    <span className="truncate text-sm">{d.name}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl px-4 py-2 text-sm font-medium text-[var(--muted)] hover:bg-foreground/5"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={busy || selected.size === 0}
            className="rounded-xl bg-orange-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-orange-400 disabled:opacity-40"
          >
            {busy ? "Adding…" : selected.size ? `Add ${selected.size}` : "Add"}
          </button>
        </div>
      </div>
    </GlassDialog>
  );
}
