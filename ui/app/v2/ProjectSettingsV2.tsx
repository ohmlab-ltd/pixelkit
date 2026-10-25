"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { API_BASE, apiFetch } from "@/lib/apiFetch";
import { patchProjectMeta } from "../../lib/projectMetaCache";
import { buildProjectLabelColourMap, LABEL_COLOURS, colourForLabelStable } from "./labelColours";

// Project settings: name, input shape, label colours, cover, snapshots, delete.

type Source = { kind: "reference" | "import"; filename: string; preview: string };

export function ProjectSettingsV2({
  projectId,
  projectName,
  labels,
  labelAliases,
  labelColours,
  references,
  imports,
  inputShape,
  inputShapeOptions,
  onInputShapeChange,
  onClose,
  onRenamed,
  onLabelColoursChange,
  onCoverChange,
  onDeleted,
}: {
  projectId: string;
  projectName: string;
  labels: string[];
  labelAliases: Record<string, string>;
  labelColours: Record<string, string>;
  references: { filename: string; preview: string }[];
  imports: { filename: string; preview: string }[];
  /** Model input size such as "256x256". The section renders only when
      all three input-shape props are set. */
  inputShape?: string;
  inputShapeOptions?: readonly string[];
  onInputShapeChange?: (next: string) => void;
  onClose: () => void;
  onRenamed: (next: string) => void;
  onLabelColoursChange: (next: Record<string, string>) => void;
  onCoverChange?: (cover: string | null) => void;
  onDeleted: () => void;
}) {
  const [newName, setNewName] = useState(projectName);
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  const [cover, setCover] = useState<string | null>(null);
  // The cover is an uploaded image rather than a picked one.
  const [coverUploaded, setCoverUploaded] = useState(false);
  const [coverBusy, setCoverBusy] = useState(false);
  // Cache-buster: cover_thumb keeps its URL across uploads.
  const [coverBust, setCoverBust] = useState(0);
  const coverFileRef = useRef<HTMLInputElement>(null);

  const [editColours, setEditColours] = useState<Record<string, string>>(labelColours ?? {});
  const [coloursError, setColoursError] = useState<string | null>(null);

  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showDelete, setShowDelete] = useState(false);

  // Loads the cover, which the parent does not hold.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await apiFetch(`/api/v2/projects/${projectId}/overview?imports_limit=0`);
        if (!r.ok) return;
        const m = await r.json() as {
          cover?: string | null;
          cover_uploaded?: boolean;
        };
        if (!alive) return;
        setCover(m.cover ?? null);
        setCoverUploaded(!!m.cover_uploaded);
      } catch {
        /* leave fields blank on error */
      }
    })();
    return () => { alive = false; };
  }, [projectId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const displayLabel = useCallback(
    (canonical: string): string => {
      const k = canonical.trim().toLowerCase();
      return labelAliases[k] || canonical;
    },
    [labelAliases],
  );

  // Colours without overrides, which Reset restores.
  const autoMap = useMemo(
    () => buildProjectLabelColourMap(labels),
    [labels],
  );
  const autoColour = (label: string): string => {
    const k = label.trim().toLowerCase();
    return autoMap.get(k) ?? colourForLabelStable(label);
  };

  const effective = (label: string): string => {
    const k = label.trim().toLowerCase();
    return editColours[k] || autoColour(label);
  };

  const rename = async () => {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === projectName) { onClose(); return; }
    setRenaming(true);
    setRenameError(null);
    try {
      const r = await apiFetch(`/api/projects/${projectId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!r.ok) {
        let msg = `http ${r.status}`;
        const body = await r.text();
        try {
          const parsed = JSON.parse(body);
          if (parsed?.detail) msg = String(parsed.detail);
          else if (body) msg = body;
        } catch { if (body) msg = body; }
        setNewName(projectName);
        setRenameError(msg);
        return;
      }
      const data = await r.json();
      onRenamed(data.name);
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId, name: data.name },
      }));
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  const setProjectCover = async (filename: string | null) => {
    if (coverBusy) return;
    const prev = cover;
    const prevUploaded = coverUploaded;
    setCover(filename);
    setCoverUploaded(false); // a pick or reset replaces an uploaded cover
    setCoverBusy(true);
    try {
      const r = await apiFetch(`/api/projects/${projectId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cover: filename }),
      });
      if (!r.ok) throw new Error(await r.text() || `http ${r.status}`);
      onCoverChange?.(filename);
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId, cover: filename },
      }));
    } catch {
      setCover(prev);
      setCoverUploaded(prevUploaded);
    } finally {
      setCoverBusy(false);
    }
  };

  // An uploaded cover replaces any picked one.
  const uploadCover = async (files: FileList | null) => {
    const f = files?.[0];
    if (!f || coverBusy) return;
    setCoverBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const r = await apiFetch(`/api/projects/${projectId}/cover_upload`, {
        method: "POST",
        body: fd,
      });
      if (!r.ok) throw new Error(await r.text() || `http ${r.status}`);
      setCover(null);
      setCoverUploaded(true);
      setCoverBust(Date.now());
      onCoverChange?.(null);
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId },
      }));
    } catch {
      /* leave the existing cover in place on failure */
    } finally {
      setCoverBusy(false);
      if (coverFileRef.current) coverFileRef.current.value = "";
    }
  };

  // Each pick saves at once and aborts any save in flight. A failed save
  // rolls back to the last saved colours.
  const lastSavedColoursRef = useRef<Record<string, string>>(labelColours ?? {});
  const saveAbortRef = useRef<AbortController | null>(null);

  const applyColoursOptimistic = useCallback((next: Record<string, string>) => {
    onLabelColoursChange(next);
    window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
      detail: { projectId, labelColours: next },
    }));
    // The meta cache gives the next page load the new colours on first paint.
    patchProjectMeta(projectId, { labelColours: next });
  }, [projectId, onLabelColoursChange]);

  const saveColours = useCallback(async (next: Record<string, string>) => {
    saveAbortRef.current?.abort();
    const ctl = new AbortController();
    saveAbortRef.current = ctl;
    setColoursError(null);
    try {
      const r = await apiFetch(`/api/projects/${projectId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labelColours: next }),
        signal: ctl.signal,
      });
      if (ctl.signal.aborted) return;
      if (!r.ok) throw new Error(await r.text() || `http ${r.status}`);
      lastSavedColoursRef.current = next;
    } catch (e) {
      // An abort means a newer pick superseded this save.
      if ((e as { name?: string })?.name === "AbortError") return;
      if (ctl.signal.aborted) return;
      setColoursError(e instanceof Error ? e.message : String(e));
      const rollback = lastSavedColoursRef.current;
      setEditColours(rollback);
      applyColoursOptimistic(rollback);
    }
  }, [projectId, applyColoursOptimistic]);

  const pickColour = (label: string, hex: string) => {
    const k = label.trim().toLowerCase();
    const next: Record<string, string> = { ...editColours };
    if (!hex) delete next[k];
    else next[k] = hex.toLowerCase();
    setEditColours(next);
    applyColoursOptimistic(next);
    void saveColours(next);
  };

  const resetColour = (label: string) => {
    const k = label.trim().toLowerCase();
    if (!(k in editColours)) return;
    const next: Record<string, string> = { ...editColours };
    delete next[k];
    setEditColours(next);
    applyColoursOptimistic(next);
    void saveColours(next);
  };

  const deleteProject = async () => {
    if (deleteBusy) return;
    if (deleteConfirm.trim() !== projectName) {
      setDeleteError("Type the dataset name exactly to confirm.");
      return;
    }
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      const r = await apiFetch(`/api/projects/${projectId}`, { method: "DELETE" });
      if (!r.ok) throw new Error(await r.text() || `http ${r.status}`);
      window.dispatchEvent(new CustomEvent("pixelkit-project-deleted", {
        detail: { projectId },
      }));
      onDeleted();
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleteBusy(false);
    }
  };

  const sources: Source[] = useMemo(() => {
    const out: Source[] = [];
    for (const r of references) out.push({ kind: "reference", filename: r.filename, preview: r.preview });
    for (const i of imports) out.push({ kind: "import", filename: i.filename, preview: i.preview });
    return out;
  }, [references, imports]);

  if (typeof window === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[600] backdrop-blur-md bg-[rgb(var(--background-rgb)/0.78)] flex items-start justify-center overflow-auto p-6"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="bg-[var(--background)] rounded-md border border-[var(--modal-border)] max-w-3xl w-full mt-8 mb-8"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-6 py-4 border-b border-foreground/10">
          <h2 className="text-[15px] font-medium text-[var(--foreground)]">Dataset settings</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-2xl leading-none px-2 text-foreground/55 hover:text-foreground"
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {/* Name */}
        <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
          <label className="pk-micro">Name</label>
          <div className="flex gap-3 items-center">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void rename(); }}
              className="flex-1 rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] focus:outline-none focus:border-[var(--line-strong)]"
            />
            <button
              type="button"
              onClick={() => void rename()}
              disabled={renaming || !newName.trim() || newName.trim() === projectName}
              className="pk-btn"
            >
              {renaming ? "Renaming…" : "Rename"}
            </button>
          </div>
          {renameError && <div className="text-xs text-[var(--bad)]">{renameError}</div>}
        </section>

        {/* Target input shape */}
        {inputShape != null && inputShapeOptions && onInputShapeChange && (
          <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
            <label className="pk-micro">Target input shape</label>
            <div className="flex flex-wrap items-center gap-3">
              <div className="inline-flex rounded-md border border-[var(--line)] p-0.5">
                {inputShapeOptions.map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => onInputShapeChange(opt)}
                    className={[
                      "rounded px-3 py-1 font-mono text-xs tabular-nums transition-colors",
                      inputShape === opt
                        ? "bg-[var(--surface-hover)] text-[var(--foreground)]"
                        : "text-foreground/60 hover:text-foreground",
                    ].join(" ")}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
            <p className="max-w-lg text-[11px] leading-relaxed text-foreground/45">
              The square tensor size your detector will resize images to before
              running. Pick the shape that matches the runtime you&rsquo;re
              targeting. The editor and export use it to flag boxes that
              would shrink below the model&rsquo;s detection floor.
            </p>
          </section>
        )}

        {/* Label colours */}
        {labels.length > 0 && (
          <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
            <label className="pk-micro">Label colours</label>
            {coloursError && (
              <div className="text-xs text-[var(--bad)]">{coloursError}</div>
            )}
            <ul className="grid gap-2">
              {labels.map((lab) => {
                const k = lab.trim().toLowerCase();
                const isOverridden = !!editColours[k];
                const eff = effective(lab);
                return (
                  <LabelColourRow
                    key={k}
                    displayName={displayLabel(lab)}
                    effectiveColour={eff}
                    isOverridden={isOverridden}
                    overrideValue={editColours[k]}
                    onPick={(hex) => pickColour(lab, hex)}
                    onReset={() => resetColour(lab)}
                  />
                );
              })}
            </ul>
            <p className="text-[11px] text-foreground/30">
              Changes show in the dataset gallery, image viewer and workspace card without a refresh.
            </p>
          </section>
        )}

        {/* Cover */}
        <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
          <div className="flex items-center justify-between">
            <label className="pk-micro">Cover image</label>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => coverFileRef.current?.click()}
                disabled={coverBusy}
                className="text-[10px] uppercase tracking-wider font-mono text-foreground/55 hover:text-foreground transition-colors disabled:opacity-40"
              >
                Upload image
              </button>
              {(cover || coverUploaded) && (
                <button
                  type="button"
                  onClick={() => setProjectCover(null)}
                  className="text-[10px] uppercase tracking-wider text-foreground/40 hover:text-foreground"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
          <input
            ref={coverFileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => uploadCover(e.target.files)}
          />
          {/* Uploaded cover */}
          {coverUploaded && (
            <div className="flex items-center gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] p-2.5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API_BASE}/api/projects/${projectId}/cover_thumb?v=${coverBust}`}
                alt="Uploaded cover"
                className="h-14 w-20 shrink-0 rounded-md object-cover"
              />
              <div className="min-w-0 text-xs leading-snug text-foreground/70">
                <span className="font-medium text-foreground/90">Using an uploaded cover.</span>{" "}
                Upload another, or pick a dataset image below to replace it.
              </div>
            </div>
          )}
          {sources.length === 0 ? (
            <p className="text-sm text-foreground/45">
              {coverUploaded ? "No dataset images yet to pick from." : "Upload a cover image, or add references or images first."}
            </p>
          ) : (
            <div className="max-h-[40vh] overflow-auto pr-1">
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                {sources.map((s) => {
                  const active = cover === s.filename;
                  return (
                    <button
                      key={`${s.kind}:${s.filename}`}
                      type="button"
                      disabled={coverBusy}
                      onClick={() => setProjectCover(s.filename)}
                      className={[
                        "relative aspect-square rounded-md overflow-hidden border-2 transition-colors",
                        active ? "border-[var(--foreground)]" : "border-transparent hover:border-foreground/40",
                        // No cursor-wait: the cursor repaints only on the
                        // next pointer event.
                        coverBusy ? "opacity-60" : "",
                      ].join(" ")}
                      title={`${s.kind}: ${s.filename}`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={s.preview}
                        alt=""
                        className="w-full h-full object-cover"
                        loading="lazy"
                        decoding="async"
                      />
                      <div className="absolute top-1 left-1 rounded-full bg-black/70 text-white/90 text-[9px] uppercase tracking-wider font-mono px-1.5 py-0.5">
                        {s.kind === "import" ? "dataset" : s.kind}
                      </div>
                      {active && (
                        <div className="absolute bottom-1 right-1 rounded-full bg-foreground text-background text-[10px] font-mono px-1.5 py-0.5">
                          cover
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>

        {/* Snapshots */}
        <SnapshotsSection projectId={projectId} />

        {/* Delete */}
        <section className="px-6 py-5 grid gap-3">
          <label className="pk-micro">Delete dataset</label>
          {!showDelete ? (
            <button
              type="button"
              onClick={() => setShowDelete(true)}
              className="self-start rounded-md border border-[var(--bad)] px-4 py-2 text-sm text-[var(--bad)] transition-colors hover:bg-[var(--bad)] hover:text-[var(--background)]"
            >
              Delete this dataset
            </button>
          ) : (
            <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] p-4 grid gap-3">
              <p className="text-sm text-[var(--bad)]">
                This permanently deletes the dataset, every reference, every imported image, and every annotation. You can&rsquo;t undo this.
              </p>
              <p className="text-xs text-foreground/65">
                Type <span className="font-mono font-medium text-[var(--foreground)]">{projectName}</span> to confirm.
              </p>
              <input
                value={deleteConfirm}
                onChange={(e) => setDeleteConfirm(e.target.value)}
                placeholder={projectName}
                className="rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-foreground/30 focus:outline-none focus:border-[var(--bad)]"
              />
              {deleteError && <div className="text-xs text-[var(--bad)]">{deleteError}</div>}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => { setShowDelete(false); setDeleteConfirm(""); setDeleteError(null); }}
                  className="pk-btn"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={deleteBusy || deleteConfirm.trim() !== projectName}
                  onClick={() => void deleteProject()}
                  className="rounded-md bg-[var(--bad)] px-4 py-1.5 text-sm font-medium text-[var(--background)] hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {deleteBusy ? "Deleting…" : "Delete dataset"}
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>,
    document.body,
  );
}

// Label colour row. Hovering the colour pip or its open rail shows the palette.
function LabelColourRow({
  displayName,
  effectiveColour,
  isOverridden,
  overrideValue,
  onPick,
  onReset,
}: {
  displayName: string;
  effectiveColour: string;
  isOverridden: boolean;
  overrideValue?: string;
  onPick: (hex: string) => void;
  onReset: () => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <li className="flex items-center gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2">
      <span className="text-sm text-foreground/85 truncate min-w-[5rem]">{displayName}</span>
      <div
        className="ml-auto flex items-center"
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        {/* Palette rail. overflow-x: clip leaves scaled swatches visible
            vertically. */}
        <div
          className="grid items-center"
          style={{
            gridTemplateColumns: hover ? "1fr" : "0fr",
            transition: "grid-template-columns 380ms cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        >
          <div
            className="min-w-0"
            style={{ overflowX: "clip", overflowY: "visible" }}
          >
            <div
              className="flex items-center gap-1.5 pr-2 py-0.5"
              style={{
                transformOrigin: "right center",
                transform: hover ? "scale(1)" : "scale(0.4)",
                opacity: hover ? 1 : 0,
                transition: hover
                  ? "transform 420ms cubic-bezier(0.22, 1, 0.36, 1), opacity 360ms ease-out"
                  : "transform 320ms cubic-bezier(0.4, 0, 0.7, 0.2), opacity 220ms ease-in",
              }}
              aria-hidden={!hover}
            >
              {LABEL_COLOURS.map((c) => {
                const active = c.toLowerCase() === effectiveColour.toLowerCase();
                return (
                  <button
                    key={c}
                    type="button"
                    aria-label={`Set ${displayName} to ${c}`}
                    onClick={() => onPick(c)}
                    className={[
                      "h-5 w-5 rounded-full border transition-transform shrink-0",
                      active ? "border-[var(--foreground)] scale-110" : "border-foreground/15 hover:scale-110 hover:border-foreground/40",
                    ].join(" ")}
                    style={{ backgroundColor: c }}
                  />
                );
              })}
              <input
                type="color"
                value={isOverridden ? (overrideValue ?? effectiveColour) : effectiveColour}
                onChange={(e) => onPick(e.target.value)}
                className="ml-1 h-5 w-5 cursor-pointer rounded-full border border-foreground/15 bg-transparent p-0 shrink-0"
                title="Pick any colour"
              />
              {isOverridden && (
                <button
                  type="button"
                  onClick={onReset}
                  className="ml-1 text-[10px] uppercase tracking-wider text-foreground/40 hover:text-foreground shrink-0"
                  title="Reset to auto-assigned colour"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>
        {/* Current colour pip */}
        <span
          aria-hidden
          className="inline-block h-5 w-5 rounded-full shrink-0 border border-foreground/15 ml-1 cursor-pointer"
          style={{ backgroundColor: effectiveColour }}
        />
      </div>
    </li>
  );
}

// Snapshots: copies of labels and annotations, without images. A restore
// snapshots the current state first.

type Snapshot = { id: string; createdAt: string; sizeBytes: number };

function fmtSnapBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

function SnapshotsSection({ projectId }: { projectId: string }) {
  const [snaps, setSnaps] = useState<Snapshot[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // "create" | snapshot id
  const [armed, setArmed] = useState<string | null>(null); // restore confirm state
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/snapshots`);
      if (r.ok) setSnaps(((await r.json()) as { snapshots: Snapshot[] }).snapshots);
    } catch {
      /* engine unreachable: the section stays empty */
    }
  }, [projectId]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const create = async () => {
    setBusy("create");
    setError(null);
    setNote(null);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/snapshots`, { method: "POST" });
      if (!r.ok) throw new Error(`Snapshot failed (${r.status}).`);
      setNote("Snapshot saved.");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const restore = async (id: string) => {
    if (armed !== id) {
      setArmed(id);
      window.setTimeout(() => setArmed((cur) => (cur === id ? null : cur)), 4000);
      return;
    }
    setArmed(null);
    setBusy(id);
    setError(null);
    setNote(null);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/snapshots/${id}/restore`, {
        method: "POST",
      });
      if (!r.ok) throw new Error(`Restore failed (${r.status}).`);
      setNote("Restored. Reloading…");
      // Annotations changed under every open view, so the page reloads.
      window.setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  };

  const fmtDate = (iso: string): string => {
    const d = new Date(iso);
    return isNaN(d.getTime()) ? iso : d.toLocaleString();
  };

  return (
    <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
      <div className="flex items-center justify-between">
        <label className="pk-micro">Annotation snapshots</label>
        <button
          type="button"
          onClick={create}
          disabled={busy !== null}
          className="text-[10px] uppercase tracking-wider font-mono text-foreground/55 hover:text-foreground transition-colors disabled:opacity-40"
        >
          {busy === "create" ? "Saving…" : "Snapshot now"}
        </button>
      </div>
      <p className="text-xs text-foreground/50">
        Labels and annotations only; images aren&rsquo;t duplicated. Restoring
        snapshots the current state first, so it&rsquo;s always undoable.
      </p>
      {(note || error) && (
        <p className={`text-xs ${error ? "text-[var(--bad)]" : "text-[var(--ok)]"}`}>
          {error ?? note}
        </p>
      )}
      {snaps && snaps.length > 0 && (
        <ul className="grid gap-1.5 max-h-48 overflow-auto pr-1">
          {snaps.map((s) => (
            <li
              key={s.id}
              className="flex items-center justify-between gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2"
            >
              <span className="min-w-0 truncate text-xs text-foreground/75 tabular-nums">
                {fmtDate(s.createdAt)}
                <span className="ml-2 text-foreground/40">{fmtSnapBytes(s.sizeBytes)}</span>
              </span>
              <button
                type="button"
                onClick={() => restore(s.id)}
                disabled={busy !== null}
                className={[
                  "shrink-0 rounded-md border px-2.5 py-1 text-[10px] uppercase tracking-wider font-mono transition-colors disabled:opacity-40",
                  armed === s.id
                    ? "border-[var(--warn)] text-[var(--warn)]"
                    : "border-[var(--line)] text-foreground/55 hover:border-[var(--line-strong)] hover:text-foreground",
                ].join(" ")}
              >
                {busy === s.id ? "Restoring…" : armed === s.id ? "Confirm restore" : "Restore"}
              </button>
            </li>
          ))}
        </ul>
      )}
      {snaps && snaps.length === 0 && (
        <p className="text-sm text-foreground/45">No snapshots yet.</p>
      )}
    </section>
  );
}
