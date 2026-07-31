"use client";

// Type-to-confirm modal for wiping every detection + edited box off
// every image in the project. Project-level labels (tags / colours)
// are preserved, only the per-image annotation data is reset.

import { useEffect, useRef, useState } from "react";

const CONFIRM_PHRASE = "I want to delete all annotations";

export function ClearAllAnnotationsModal({
  onCancel,
  onConfirm,
}: {
  onCancel: () => void;
  onConfirm: () => Promise<void> | void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const t = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(t);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, busy]);

  const matches = typed === CONFIRM_PHRASE;
  const submit = async () => {
    if (!matches || busy) return;
    setBusy(true);
    try {
      await onConfirm();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="pk-backdrop fixed inset-0 z-[1300] grid place-items-center p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Clear all annotations"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel();
      }}
    >
      <div className="pk-glass pk-pop w-full max-w-md rounded-2xl p-6">
        <h2 className="text-2xl font-medium tracking-tight text-[var(--foreground)]">
          Clear all annotations?
        </h2>
        <p className="mt-3 text-sm text-foreground/65 leading-relaxed">
          This permanently removes every detection and edited box from{" "}
          <span className="font-medium text-[var(--foreground)]">every image</span>{" "}
          in this project. Labels themselves stay, you can re-label without
          reconfiguring. This cannot be undone.
        </p>
        <p className="mt-4 text-[12px] text-foreground/55">
          Type{" "}
          <span className="font-mono font-semibold text-[var(--foreground)]">
            {CONFIRM_PHRASE}
          </span>
          {" "}to confirm.
        </p>
        <input
          ref={inputRef}
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); void submit(); }
          }}
          placeholder={CONFIRM_PHRASE}
          disabled={busy}
          className="mt-2 w-full rounded-xl border border-foreground/15 bg-foreground/[0.02] px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-foreground/30 focus:outline-none focus:border-foreground/35 disabled:opacity-60"
        />
        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-foreground/15 px-4 py-2 text-sm text-foreground/75 hover:bg-foreground/[0.04] hover:text-foreground transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => void submit()}
            disabled={!matches || busy}
            className="rounded-full border border-rose-500/45 bg-rose-500/[0.10] px-4 py-2 text-sm font-medium text-rose-700 dark:text-rose-200 hover:bg-rose-500/[0.18] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Clearing…" : "Clear all annotations"}
          </button>
        </div>
      </div>
    </div>
  );
}
