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
          <span className="text-sm font-medium text-foreground/80">Dataset name</span>
          <input
            ref={nameRef}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            maxLength={120}
            placeholder="e.g. Rooftop panels"
            className="rounded-xl border border-foreground/10 bg-foreground/[0.03] px-3.5 py-2.5 text-sm outline-none focus:border-orange-400 focus:ring-2 focus:ring-orange-400/30"
          />
        </label>
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
            disabled={!name.trim()}
            className="rounded-xl bg-orange-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-orange-400 disabled:opacity-40"
          >
            Continue
          </button>
        </div>
      </div>
    </GlassDialog>
  );
}
