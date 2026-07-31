"use client";

// Create-a-Project (container) modal: name + cover photo + privacy. Calls the
// container backend via lib/containers. A Project holds many Datasets and has
// members; this is distinct from creating a Dataset (the CreateFlowV2 path).
import { useEffect, useRef, useState } from "react";

import { GlassDialog } from "./v2/GlassDialog";
import { createContainer, uploadCover, type Container } from "@/lib/containers";

export function CreateProjectModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (c: Container) => void;
}) {
  const [name, setName] = useState("");
  const [isPrivate, setIsPrivate] = useState(true);
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);

  // Focus the name field when the modal opens. GlassDialog focuses its first
  // control (the cover button) on open; this parent effect runs after that
  // child effect, and the deferred tick makes it win, so the user can type
  // straight away.
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => nameRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  function pickCover(f: File | null) {
    if (coverPreview) URL.revokeObjectURL(coverPreview);
    setCoverFile(f);
    setCoverPreview(f ? URL.createObjectURL(f) : null);
  }

  function reset() {
    setName("");
    setIsPrivate(true);
    pickCover(null);
    setError(null);
    setBusy(false);
  }

  function handleClose() {
    if (busy) return;
    reset();
    onClose();
  }

  async function submit() {
    const nm = name.trim();
    if (!nm) {
      setError("Give your project a name.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const c = await createContainer(nm, isPrivate);
      if (!c) {
        setError("Could not create the project. Please try again.");
        setBusy(false);
        return;
      }
      if (coverFile) {
        await uploadCover(c.id, coverFile);
      }
      reset();
      onCreated(c);
    } catch {
      setError("Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  return (
    <GlassDialog open={open} onClose={handleClose} title="New project" maxWidth="max-w-md">
      <div className="flex flex-col gap-5">
        <p className="text-sm text-[var(--muted)] -mt-1">
          A project groups datasets and lets your team collaborate. You can add datasets and members
          once it&apos;s created.
        </p>

        {/* Cover photo */}
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="group relative aspect-[16/9] w-full overflow-hidden rounded-2xl border border-foreground/10 bg-foreground/[0.03] transition hover:border-orange-400/60"
        >
          {coverPreview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={coverPreview} alt="" className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-foreground/40">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                <rect x="3" y="3" width="18" height="18" rx="3" />
                <circle cx="8.5" cy="8.5" r="1.6" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
              <span className="text-sm font-medium">Upload a cover photo</span>
            </div>
          )}
          <span className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/40 to-transparent px-3 py-2 text-xs font-medium text-white opacity-0 transition group-hover:opacity-100">
            {coverPreview ? "Change cover" : "Recommended: a wide image"}
          </span>
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => pickCover(e.target.files?.[0] ?? null)}
        />

        {/* Name */}
        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-foreground/80">Project name</span>
          <input
            ref={nameRef}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            maxLength={120}
            placeholder="e.g. Retail shelf detection"
            className="rounded-xl border border-foreground/10 bg-foreground/[0.03] px-3.5 py-2.5 text-sm outline-none focus:border-orange-400 focus:ring-2 focus:ring-orange-400/30"
          />
        </label>

        {/* Privacy */}
        <div className="flex items-center justify-between rounded-xl border border-foreground/10 px-3.5 py-3">
          <div className="flex flex-col">
            <span className="text-sm font-medium text-foreground/85">
              {isPrivate ? "Private" : "Public"}
            </span>
            <span className="text-xs text-[var(--muted)]">
              {isPrivate
                ? "Only you and members can see it and its datasets."
                : "Anyone can view it and its datasets."}
            </span>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={isPrivate}
            onClick={() => setIsPrivate((p) => !p)}
            // The switch represents "Private": ON (orange, knob right) = private,
            // OFF = public. Defaults to ON since new projects start private.
            className={`relative h-6 w-11 shrink-0 rounded-full transition ${
              isPrivate ? "bg-orange-500" : "bg-foreground/20"
            }`}
          >
            <span
              className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${
                isPrivate ? "left-[22px]" : "left-0.5"
              }`}
            />
          </button>
        </div>

        {error && <p className="text-sm text-rose-500">{error}</p>}

        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={handleClose}
            disabled={busy}
            className="rounded-xl px-4 py-2 text-sm font-medium text-[var(--muted)] hover:bg-foreground/5 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={busy || !name.trim()}
            className="rounded-xl bg-orange-500 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "Creating…" : "Create project"}
          </button>
        </div>
      </div>
    </GlassDialog>
  );
}
