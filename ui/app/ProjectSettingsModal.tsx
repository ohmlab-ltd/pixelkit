"use client";

// Project (container) settings pop-out: rename, cover, privacy (with an
// are-you-sure confirm because it cascades to every dataset), image quality,
// and delete. Owner-only actions; the modal is only opened for owners. The
// portable build is single-user, so there is no members section (accounts /
// identity are invisible everywhere).
import { useRef, useState } from "react";

import { GlassDialog } from "./v2/GlassDialog";
import {
  deleteContainer,
  patchContainer,
  uploadCover,
  type ContainerDetail,
} from "@/lib/containers";

export function ProjectSettingsModal({
  container,
  open,
  onClose,
  onChanged,
  onDeleted,
}: {
  container: ContainerDetail;
  open: boolean;
  onClose: () => void;
  onChanged: () => void;
  /** Called after the Project is deleted (parent navigates back). */
  onDeleted?: () => void;
}) {
  const [name, setName] = useState(container.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Privacy confirm flow.
  const [confirmPrivacy, setConfirmPrivacy] = useState(false);
  // Delete confirm flow.
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Max input image size (px longest edge for uploads). Datasets inherit it.
  const [inputSize, setInputSize] = useState<number>(container.max_input_size ?? 1500);

  async function saveName() {
    const nm = name.trim();
    if (!nm || nm === container.name) return;
    setBusy(true);
    setError(null);
    if (await patchContainer(container.id, { name: nm })) onChanged();
    else setError("Could not rename the project.");
    setBusy(false);
  }

  async function changeCover(file: File | null) {
    if (!file) return;
    setBusy(true);
    setError(null);
    if (await uploadCover(container.id, file)) onChanged();
    else setError("Could not update the cover.");
    setBusy(false);
  }

  async function applyPrivacy() {
    setBusy(true);
    setError(null);
    setConfirmPrivacy(false);
    if (await patchContainer(container.id, { private: !container.private })) onChanged();
    else setError("Could not change privacy.");
    setBusy(false);
  }

  async function applyInputSize(v: number) {
    setInputSize(v);
    setBusy(true);
    setError(null);
    if (await patchContainer(container.id, { max_input_size: v })) onChanged();
    else setError("Could not change the image quality.");
    setBusy(false);
  }

  async function doDelete() {
    setBusy(true);
    setError(null);
    if (await deleteContainer(container.id)) {
      onDeleted?.();
    } else {
      setError("Could not delete the project.");
      setBusy(false);
    }
  }

  return (
    <GlassDialog open={open} onClose={onClose} title="Project settings" maxWidth="max-w-lg">
      <div className="flex flex-col gap-6">
        {/* Name */}
        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-foreground/80">Name</span>
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={120}
              className="flex-1 rounded-xl border border-foreground/10 bg-foreground/[0.03] px-3.5 py-2.5 text-sm outline-none focus:border-orange-400 focus:ring-2 focus:ring-orange-400/30"
            />
            <button
              type="button"
              onClick={saveName}
              disabled={busy || !name.trim() || name.trim() === container.name}
              className="rounded-xl bg-orange-500 px-4 text-sm font-semibold text-white transition hover:bg-orange-400 disabled:opacity-40"
            >
              Save
            </button>
          </div>
        </div>

        {/* Cover */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-foreground/80">Cover photo</span>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={busy}
            className="rounded-xl border border-foreground/10 px-4 py-2 text-sm font-medium hover:bg-foreground/5 disabled:opacity-50"
          >
            Change cover
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => changeCover(e.target.files?.[0] ?? null)}
          />
        </div>

        {/* Privacy with confirm */}
        <div className="rounded-xl border border-foreground/10 p-3.5">
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-sm font-medium text-foreground/85">
                {container.private ? "Private" : "Public"}
              </span>
              <span className="text-xs text-[var(--muted)]">
                Changing this also changes every dataset in the project.
              </span>
            </div>
            {!confirmPrivacy ? (
              <button
                type="button"
                onClick={() => setConfirmPrivacy(true)}
                disabled={busy}
                className="rounded-xl border border-foreground/10 px-4 py-2 text-sm font-medium hover:bg-foreground/5 disabled:opacity-50"
              >
                Make {container.private ? "public" : "private"}
              </button>
            ) : null}
          </div>
          {confirmPrivacy && (
            <div className="mt-3 flex items-center justify-between gap-2 rounded-lg bg-amber-500/10 px-3 py-2">
              <span className="text-xs text-amber-700 dark:text-amber-300">
                Are you sure? This makes the project and all its datasets{" "}
                {container.private ? "public" : "private"}.
              </span>
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmPrivacy(false)}
                  className="rounded-lg px-3 py-1 text-xs font-medium text-[var(--muted)]"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={applyPrivacy}
                  disabled={busy}
                  className="rounded-lg bg-amber-500 px-3 py-1 text-xs font-semibold text-white disabled:opacity-50"
                >
                  Yes, change it
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Max input image size */}
        <div className="flex items-center justify-between gap-3 rounded-xl border border-foreground/10 p-3.5">
          <div className="flex flex-col">
            <span className="text-sm font-medium text-foreground/85">Image quality</span>
            <span className="text-xs text-[var(--muted)]">
              Longest edge new uploads are resized to. Higher keeps more detail and uses more storage.
            </span>
          </div>
          <select
            value={inputSize}
            onChange={(e) => applyInputSize(Number(e.target.value))}
            disabled={busy}
            className="shrink-0 rounded-xl border border-foreground/10 bg-foreground/[0.03] px-2.5 py-2 text-sm"
          >
            <option value={1024}>Compact (1024px)</option>
            <option value={1500}>Standard (1500px)</option>
            <option value={2048}>High (2048px)</option>
            <option value={4096}>4K (4096px)</option>
          </select>
        </div>

        {/* Danger zone */}
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.04] p-3.5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex flex-col">
              <span className="text-sm font-medium text-foreground/85">Delete project</span>
              <span className="text-xs text-[var(--muted)]">
                The datasets are kept (they become standalone).
              </span>
            </div>
            {!confirmDelete && (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                disabled={busy}
                className="shrink-0 rounded-xl border border-rose-500/40 px-4 py-2 text-sm font-medium text-rose-600 transition hover:bg-rose-500/10 disabled:opacity-50 dark:text-rose-400"
              >
                Delete
              </button>
            )}
          </div>
          {confirmDelete && (
            <div className="mt-3 flex items-center justify-between gap-2 rounded-lg bg-rose-500/10 px-3 py-2">
              <span className="text-xs text-rose-700 dark:text-rose-300">
                Delete &ldquo;{container.name}&rdquo;? This can&apos;t be undone.
              </span>
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="rounded-lg px-3 py-1 text-xs font-medium text-[var(--muted)]"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={doDelete}
                  disabled={busy}
                  className="rounded-lg bg-rose-600 px-3 py-1 text-xs font-semibold text-white transition hover:bg-rose-500 disabled:opacity-50"
                >
                  Yes, delete
                </button>
              </div>
            </div>
          )}
        </div>

        {error && <p className="text-sm text-rose-500">{error}</p>}
      </div>
    </GlassDialog>
  );
}
