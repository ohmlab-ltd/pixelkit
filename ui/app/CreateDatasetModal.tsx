"use client";

// Collect a name for a new dataset created from INSIDE a Project, then hand off
// to the full V2 onboarding (labels -> references -> upload). The name is
// entered here, within the project, rather than jumping to the workspace; the
// parent then animates the project away and starts the onboarding.
import { useEffect, useRef, useState } from "react";

import { GlassDialog } from "./v2/GlassDialog";

export function CreateDatasetModal({
  open,
  onClose,
  onContinue,
}: {
  open: boolean;
  onClose: () => void;
  onContinue: (name: string) => void;
}) {
  const [name, setName] = useState("");
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setName("");
    const t = setTimeout(() => nameRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  function submit() {
    const nm = name.trim();
    if (!nm) return;
    onContinue(nm);
  }

  return (
    <GlassDialog open={open} onClose={onClose} title="New dataset" maxWidth="max-w-md">
      <div className="flex flex-col gap-5">
        <p className="-mt-1 text-sm text-[var(--muted)]">
          Name your dataset. Next you&apos;ll add labels and images.
        </p>
        <label className="flex flex-col gap-1.5">
          <span className="pk-micro">Dataset name</span>
          <input
            ref={nameRef}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            maxLength={120}
            placeholder="e.g. Rooftop panels"
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-sm outline-none transition-colors focus:border-[var(--accent)]"
          />
        </label>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-[var(--line)] px-4 py-2 text-[13px] font-medium text-foreground/75 transition-colors hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={!name.trim()}
            className="rounded-md bg-[var(--accent)] px-4 py-2 text-[13px] font-medium text-[var(--accent-contrast)] transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Continue
          </button>
        </div>
      </div>
    </GlassDialog>
  );
}
