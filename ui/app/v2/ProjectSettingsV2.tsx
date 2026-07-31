"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiFetch } from "@/lib/apiFetch";
import { isProPlan } from "@/lib/plans";
import { patchProjectMeta } from "../../lib/projectMetaCache";
import { usePlan } from "../PlanPill";
import { containsProfanity } from "../profanity";
import { buildProjectLabelColourMap, LABEL_COLOURS, colourForLabelStable } from "./OnboardLabelsV2";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// V2-native settings popup. Five sections:
//   1. Rename           , POST /api/projects/{id}/rename
//   2. Visibility       , PUT /api/projects/{id} {private}; gated on usePlan().plan
//   3. Cover            , PUT /api/projects/{id} {cover}; picks from refs OR imports
//   4. Label colours    , PUT /api/projects/{id} {labelColours}; live-broadcasts
//   5. Delete           , DELETE /api/projects/{id} {confirm: <name>}
//
// Live propagation: on every successful save we mirror state through
// the existing window-event channel so the workspace + public cards
// repaint without waiting for their next poll.

type Source = { kind: "reference" | "import"; filename: string; preview: string };

export function ProjectSettingsV2({
  projectId,
  projectName,
  initialPrivate,
  labels,
  labelAliases,
  labelColours,
  references,
  imports,
  onClose,
  onRenamed,
  onLabelColoursChange,
  onCoverChange,
  onPrivateChange,
  onDeleted,
}: {
  projectId: string;
  projectName: string;
  initialPrivate?: boolean;
  labels: string[];
  labelAliases: Record<string, string>;
  labelColours: Record<string, string>;
  references: { filename: string; preview: string }[];
  imports: { filename: string; preview: string }[];
  onClose: () => void;
  onRenamed: (next: string) => void;
  onLabelColoursChange: (next: Record<string, string>) => void;
  onCoverChange?: (cover: string | null) => void;
  onPrivateChange?: (next: boolean) => void;
  onDeleted: () => void;
}) {
  const plan = usePlan();
  const planTier = plan?.plan ?? "free";
  // Beta accounts get the same surface area as Pro for the duration
  // of their access window, including private projects, so the
  // toggle is gated on any non-free plan.
  const canPrivate =
    isProPlan(planTier) || planTier === "mega" || planTier === "beta" || planTier === "enterprise";

  const [newName, setNewName] = useState(projectName);
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  const [isPrivate, setIsPrivate] = useState<boolean>(initialPrivate ?? false);
  const [privacyBusy, setPrivacyBusy] = useState(false);
  const [privacyError, setPrivacyError] = useState<string | null>(null);
  // Tracks whether the user has already clicked the toggle this session.
  // If so, we skip the useEffect hydration write to avoid clobbering an
  // in-flight or completed optimistic update.
  const privacyInteractedRef = useRef(false);
  const toggleAbortRef = useRef<AbortController | null>(null);

  const [cover, setCover] = useState<string | null>(null);
  // True when the dataset is using a user-uploaded cover (vs one picked from an
  // import/reference). Drives the "current cover" preview + supersedes a grid pick.
  const [coverUploaded, setCoverUploaded] = useState(false);
  const [coverBusy, setCoverBusy] = useState(false);
  // Cache-buster so the uploaded-cover preview refreshes after a re-upload
  // (cover_thumb keeps the same URL across uploads).
  const [coverBust, setCoverBust] = useState(0);
  const coverFileRef = useRef<HTMLInputElement>(null);

  const [editColours, setEditColours] = useState<Record<string, string>>(labelColours ?? {});
  const [coloursError, setColoursError] = useState<string | null>(null);

  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showDelete, setShowDelete] = useState(false);

  // Hydrate fields the stub doesn't already hold in state. The parent
  // already owns labelColours / labels / aliases; we just need
  // private + cover (manifest-only fields the V2 view doesn't track).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await apiFetch(`/api/projects/${projectId}`);
        if (!r.ok) return;
        const m = await r.json() as {
          private?: boolean;
          cover?: string | null;
          cover_uploaded?: boolean;
        };
        if (!alive) return;
        if (!privacyInteractedRef.current) setIsPrivate(!!m.private);
        setCover(m.cover ?? null);
        setCoverUploaded(!!m.cover_uploaded);
      } catch {
        /* leave fields blank on error */
      }
    })();
    return () => { alive = false; };
  }, [projectId]);

  // Esc closes, same shortcut V1 ProjectSettings ships.
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

  // Per-label auto-assigned colour map. Read at render time so the
  // swatch grid shows what the project would use WITHOUT any
  // override, that's the colour the "Reset" button restores to.
  const autoMap = useMemo(
    () => buildProjectLabelColourMap(labels),
    [labels],
  );
  const autoColour = (label: string): string => {
    const k = label.trim().toLowerCase();
    return autoMap.get(k) ?? colourForLabelStable(label);
  };

  // Effective colour for the swatch row: override wins, otherwise
  // the auto-assigned palette slot. Used both to paint each label
  // row's main swatch + to highlight the active preset in the grid.
  const effective = (label: string): string => {
    const k = label.trim().toLowerCase();
    return editColours[k] || autoColour(label);
  };

  const rename = async () => {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === projectName) { onClose(); return; }
    if (containsProfanity(trimmed)) {
      setRenameError(`"${trimmed}" can't be used as a project name.`);
      setNewName(projectName);
      return;
    }
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
        if (/banned term/i.test(msg)) setRenameError(`"${trimmed}" can't be used as a project name.`);
        else setRenameError(msg);
        return;
      }
      const data = await r.json();
      onRenamed(data.name);
      // Broadcast so other open tabs / sibling lists patch their state.
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId, name: data.name },
      }));
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  const togglePrivate = async (next: boolean) => {
    if (!canPrivate) return;
    // Cancel any in-flight toggle so rapid clicks don't race.
    toggleAbortRef.current?.abort();
    const ctl = new AbortController();
    toggleAbortRef.current = ctl;

    privacyInteractedRef.current = true;
    const prev = isPrivate;

    // Optimistic update — apply immediately so the padlock and any other
    // listeners reflect the new state before the PUT even fires. The
    // modal closing mid-flight is fine because the parent already holds
    // the new value; we only roll back if the server rejects it.
    setIsPrivate(next);
    setPrivacyBusy(true);
    setPrivacyError(null);
    onPrivateChange?.(next);
    window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
      detail: { projectId, private: next },
    }));

    try {
      const r = await apiFetch(`/api/projects/${projectId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ private: next }),
        signal: ctl.signal,
      });
      if (ctl.signal.aborted) return;
      if (!r.ok) {
        const body = await r.text().catch(() => "");
        let message = body || `http ${r.status}`;
        if (r.status === 403) {
          message =
            "Only the project owner can change visibility. If you're signed in with the right account and still see this, the project may pre-date the ownership-required era; contact support to claim it.";
        }
        throw new Error(message);
      }
    } catch (e) {
      if ((e as { name?: string })?.name === "AbortError") return;
      if (ctl.signal.aborted) return;
      console.warn("[settings/private] toggle failed:", e);
      // Roll back local state, parent state, and any other listeners.
      setIsPrivate(prev);
      onPrivateChange?.(prev);
      window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
        detail: { projectId, private: prev },
      }));
      setPrivacyError(e instanceof Error ? e.message : String(e));
    } finally {
      if (!ctl.signal.aborted) setPrivacyBusy(false);
    }
  };

  const setProjectCover = async (filename: string | null) => {
    if (coverBusy) return;
    const prev = cover;
    const prevUploaded = coverUploaded;
    setCover(filename);
    setCoverUploaded(false); // picking (or resetting) supersedes an uploaded cover
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

  // Upload a custom cover image (POST multipart) and flag the dataset as using
  // an uploaded cover. Clears any grid selection since the upload wins.
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

  // Colour save, fires INSTANTLY. The previous version debounced
  // 350 ms to coalesce rapid clicks; that meant a refresh inside the
  // debounce window lost the user's pick. Now the PUT goes out on
  // every change, and an AbortController cancels any in-flight save
  // when a fresh pick lands so a fast click-through can't end with
  // an earlier request landing AFTER a later one (write-write race).
  // Last-saved snapshot drives rollback if a save comes back failed.
  const lastSavedColoursRef = useRef<Record<string, string>>(labelColours ?? {});
  const saveAbortRef = useRef<AbortController | null>(null);

  const applyColoursOptimistic = useCallback((next: Record<string, string>) => {
    // Parent state, drives chips under the heading + the dataset
    // gallery, plus the cross-window event so any other open
    // surface (terminal, public feed card) picks up the change.
    onLabelColoursChange(next);
    window.dispatchEvent(new CustomEvent("pixelkit-project-meta-changed", {
      detail: { projectId, labelColours: next },
    }));
    // Mirror into the persistent meta cache so the NEXT page load
    // paints with the new colour on the first frame, instead of
    // flashing the previous value while the /overview fetch lands.
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
      // Aborts from a superseding pick are expected, don't surface
      // them as errors or roll back valid state.
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
      setDeleteError("Type the project name exactly to confirm.");
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
      // Themable modal backdrop: tints with the active background
      // so the blur reads as a soft frosted overlay in light mode
      // instead of the hard black panel that flipped both modes.
      className="fixed inset-0 z-[600] backdrop-blur-md bg-[rgb(var(--background-rgb)/0.78)] flex items-start justify-center overflow-auto p-6"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="bg-[var(--background)] rounded-2xl border border-foreground/10 max-w-3xl w-full mt-8 mb-8"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-6 py-4 border-b border-foreground/10">
          <h2 className="text-lg font-semibold text-[var(--foreground)]">Project settings</h2>
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
          <label className="text-xs text-foreground/45 uppercase tracking-wider">Name</label>
          <div className="flex gap-3 items-center">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void rename(); }}
              className="flex-1 rounded-lg border border-foreground/10 bg-transparent px-3 py-2 text-base text-[var(--foreground)] focus:outline-none focus:border-foreground/30"
            />
            <button
              type="button"
              onClick={() => void rename()}
              disabled={renaming || !newName.trim() || newName.trim() === projectName}
              className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {renaming ? "Renaming…" : "Rename"}
            </button>
          </div>
          {renameError && <div className="text-xs text-red-400">{renameError}</div>}
        </section>

        {/* Visibility */}
        <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
          <label className="text-xs text-foreground/45 uppercase tracking-wider">Visibility</label>
          {canPrivate ? (
            <button
              type="button"
              role="switch"
              aria-checked={!!isPrivate}
              onClick={() => togglePrivate(!isPrivate)}
              className={[
                "inline-flex items-center gap-3 self-start rounded-full border px-3 py-1.5 text-sm transition-colors",
                isPrivate
                  ? "border-amber-500/50 bg-amber-300/[0.12] text-amber-800 dark:text-amber-100 hover:bg-amber-300/[0.18]"
                  : "border-foreground/20 bg-foreground/5 text-[var(--foreground)] hover:bg-foreground/10",
                privacyBusy ? "opacity-75" : "",
              ].join(" ")}
            >
              <span
                aria-hidden
                className={[
                  "h-4 w-7 rounded-full p-0.5 transition-colors flex",
                  isPrivate ? "bg-amber-500/80 justify-end" : "bg-foreground/25",
                ].join(" ")}
              >
                <span className="h-3 w-3 rounded-full bg-background" />
              </span>
              {isPrivate
                ? "Private, only you can see this project"
                : "Public, visible in the community feed"}
            </button>
          ) : null}
          {privacyError && (
            <div className="text-[12px] text-red-500 dark:text-red-300 max-w-md leading-relaxed">
              {privacyError}
            </div>
          )}
          {canPrivate ? null : (
            <div className="flex flex-col gap-2 self-start rounded-xl border border-foreground/10 bg-foreground/[0.02] px-4 py-3 text-sm text-foreground/65 max-w-md">
              <span>
                Private projects are a Pro feature. Free projects stay public in the community feed.
              </span>
              <a
                href="/app?tab=pricing"
                className="self-start text-[11px] uppercase tracking-wider font-mono text-orange-700 hover:text-orange-800 dark:text-orange-200 dark:hover:text-orange-100"
              >
                Upgrade to Pro →
              </a>
            </div>
          )}
        </section>

        {/* Label colours */}
        {labels.length > 0 && (
          <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
            <label className="text-xs text-foreground/45 uppercase tracking-wider">Label colours</label>
            {coloursError && (
              <div className="text-xs text-red-400">{coloursError}</div>
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
              Changes propagate to the dataset gallery, image viewer, workspace card, and public feed without a refresh.
            </p>
          </section>
        )}

        {/* Cover */}
        <section className="px-6 py-5 border-b border-foreground/10 grid gap-3">
          <div className="flex items-center justify-between">
            <label className="text-xs text-foreground/45 uppercase tracking-wider">Cover image</label>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => coverFileRef.current?.click()}
                disabled={coverBusy}
                className="text-[10px] uppercase tracking-wider text-[var(--accent-orange)] hover:opacity-80 disabled:opacity-40"
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
          {/* Current uploaded cover preview — sits above the pick-from-images
              grid so the user can see (and replace) it. */}
          {coverUploaded && (
            <div className="flex items-center gap-3 rounded-lg border border-[rgb(var(--accent-orange-rgb)/0.4)] bg-[rgb(var(--accent-orange-rgb)/0.06)] p-2.5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API}/api/projects/${projectId}/cover_thumb?v=${coverBust}`}
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
                        "relative aspect-square rounded-lg overflow-hidden border-2 transition-colors",
                        active ? "border-[var(--foreground)]" : "border-transparent hover:border-foreground/40",
                        // No cursor-wait: the cursor only repaints on
                        // the next pointer event, so it'd appear to
                        // stick until the user moves the mouse, even
                        // though the PUT lands instantly. opacity is
                        // enough to signal the in-flight state.
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
                      {/* Floats over the cover thumbnail, keep the
                          dark bubble + white text in both themes
                          since it has to read on arbitrary image
                          content. */}
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

        {/* Delete */}
        <section className="px-6 py-5 grid gap-3">
          <label className="text-xs text-foreground/45 uppercase tracking-wider">Delete project</label>
          {!showDelete ? (
            <button
              type="button"
              onClick={() => setShowDelete(true)}
              // Deeper red in light mode so the destructive warning
              // text doesn't wash out against the near-white card.
              className="self-start rounded-xl border border-red-500/40 bg-red-500/[0.08] px-4 py-2 text-sm text-red-700 dark:text-red-200 hover:bg-red-500/[0.14] transition-colors"
            >
              Delete this project
            </button>
          ) : (
            <div className="rounded-xl border border-red-500/40 bg-red-500/[0.06] p-4 grid gap-3">
              <p className="text-sm text-red-800 dark:text-red-100/90">
                This permanently deletes the project, every reference, every imported image, and every annotation. You can&rsquo;t undo this.
              </p>
              {/* "Type X to confirm", strong red so it stays
                  legible against the pink delete-card background
                  in light mode (foreground/55 was washing out on
                  the tinted bg). */}
              <p className="text-xs text-red-800 dark:text-red-100/80">
                Type <span className="font-mono font-semibold text-red-900 dark:text-red-50">{projectName}</span> to confirm.
              </p>
              <input
                value={deleteConfirm}
                onChange={(e) => setDeleteConfirm(e.target.value)}
                placeholder={projectName}
                className="rounded-lg border border-red-500/30 bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-foreground/30 focus:outline-none focus:border-red-500/70"
              />
              {deleteError && <div className="text-xs text-red-700 dark:text-red-300">{deleteError}</div>}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => { setShowDelete(false); setDeleteConfirm(""); setDeleteError(null); }}
                  className="rounded-full border border-foreground/15 px-3.5 py-1.5 text-xs font-medium text-foreground/65 hover:border-foreground/30 hover:text-foreground"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={deleteBusy || deleteConfirm.trim() !== projectName}
                  onClick={() => void deleteProject()}
                  // Solid destructive button with always-white text so
                  // it doesn't theme to dark-on-red in light mode.
                  className="rounded-full bg-red-600 text-white px-5 py-1.5 text-sm font-semibold hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {deleteBusy ? "Deleting…" : "Delete project"}
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

// Single row in the label-colour list. Default view shows just the
// current colour as a small circle on the right; hovering anywhere
// over the row expands the preset palette + colour picker + reset
// outwards (right-anchored), then collapses with the same animation
// when the mouse leaves the chip.
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
  // Hover-tracked on the pip+rail wrapper specifically, the row
  // itself doesn't open the picker, only the colour pip does. Stays
  // open while the cursor sits anywhere over the expanded rail.
  const [hover, setHover] = useState(false);
  return (
    <li className="flex items-center gap-3 rounded-xl border border-foreground/[0.07] bg-foreground/[0.02] px-3 py-2">
      <span className="text-sm text-foreground/85 truncate min-w-[5rem]">{displayName}</span>
      <div
        className="ml-auto flex items-center"
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        {/* Expanding rail, preset swatches + native picker + reset.
            Width interpolates via grid-template-columns; the inner
            row scales out from the pip (transform-origin: right) so
            the swatches feel like they're emerging from the bubble.
            overflow-x: clip lets the horizontal clip stay tight while
            keeping the vertical axis visible so :hover scale on
            individual swatches isn't sliced off. */}
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
        {/* Current-colour pip, anchor + hover target. Always
            visible; the rail collapses back into this circle. */}
        <span
          aria-hidden
          className="inline-block h-5 w-5 rounded-full shrink-0 border border-foreground/15 ml-1 cursor-pointer"
          style={{ backgroundColor: effectiveColour }}
        />
      </div>
    </li>
  );
}
