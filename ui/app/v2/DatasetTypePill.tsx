"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const MENU_WIDTH = 288; // matches w-72

// Compact dataset-type indicator for the top of the project page.
// Shows whether PixelKit is treating the project as a SPECIFIC dataset
// (look-alike classes — references + embedding scoring) or a GENERAL
// one (distinct categories — plain text-prompt detection), surfaces the
// short reason behind that call, and (for the owner) lets them change
// it or hand control back to the classifier. Styling mirrors the muted
// pill chrome used elsewhere in the header row.

export type DatasetTypeValue = {
  type: "general" | "specific";
  reason?: string | null;
  // "auto" = classifier, "manual" = owner override, "references" =
  // flipped to specific because reference images were added.
  source?: string | null;
};

const OPTIONS: { key: "specific" | "general" | "auto"; label: string; desc: string }[] = [
  { key: "specific", label: "Specific", desc: "Look-alike classes — uses reference images." },
  { key: "general", label: "General", desc: "Distinct categories — text prompt is enough." },
  { key: "auto", label: "Let PixelKit decide", desc: "Re-classify from the labels." },
];

export function DatasetTypePill({
  value,
  readOnly,
  onChange,
  derived = false,
  onCover = false,
  light = false,
}: {
  value: DatasetTypeValue | null;
  readOnly?: boolean;
  onChange?: (choice: "general" | "specific" | "auto") => void;
  // Derived (child) projects are a fixed special type: show a static
  // "Derived" badge with no general/specific picker.
  derived?: boolean;
  // When rendered over a cover hero, switch to a content-aware glassy pill
  // (the muted theme chrome is unreadable on an image). `light` = light cover.
  onCover?: boolean;
  light?: boolean;
}) {
  const [open, setOpen] = useState(false);
  // The menu is rendered into a portal at document.body with fixed
  // positioning so it escapes the dataset hero's `overflow-hidden` box
  // (which otherwise clips a normally-positioned absolute dropdown). We
  // measure the trigger on open and anchor the menu to it, clamping to the
  // viewport so a pill in the top-right corner doesn't push the menu
  // off-screen.
  const btnRef = useRef<HTMLButtonElement>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);

  const openMenu = () => {
    const el = btnRef.current;
    if (!el) { setOpen(true); return; }
    const r = el.getBoundingClientRect();
    const left = Math.min(Math.max(8, r.left), window.innerWidth - MENU_WIDTH - 8);
    setMenuPos({ top: r.bottom + 8, left });
    setOpen(true);
  };

  // Close on scroll/resize while open: the fixed menu would otherwise drift
  // away from its (scrolling) trigger.
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  // Glassy pill chrome when over a cover; null otherwise (keeps the original
  // muted theme styling on the standard header row).
  const coverCls = onCover
    ? light
      ? "bg-black/10 text-zinc-900 ring-1 ring-black/10 backdrop-blur-md"
      : "bg-white/15 text-white ring-1 ring-white/10 backdrop-blur-md"
    : null;
  if (derived) {
    return (
      <span
        className={`inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-[11px] font-medium uppercase tracking-wider ${coverCls ?? "bg-sky-500/[0.12] text-sky-700 dark:text-sky-300"}`}
        title="A cropped child dataset derived from a parent project"
      >
        <span className={`h-1.5 w-1.5 rounded-full ${onCover ? "bg-current opacity-70" : "bg-sky-500"}`} aria-hidden />
        Derived
      </span>
    );
  }
  if (!value) return null;

  const isSpecific = value.type === "specific";
  const dotClass = isSpecific
    ? "bg-amber-400"
    : onCover
      ? "bg-current opacity-60"
      : "bg-foreground/40";
  const label = isSpecific ? "Specific" : "General";
  const reason = value.reason?.trim() || null;

  const inner = (
    <>
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} aria-hidden />
      {label}
    </>
  );

  // Read-only viewers (public project pages) get a static badge with the
  // reason on hover — no menu.
  if (readOnly || !onChange) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 h-6 text-[11px] font-medium uppercase tracking-wider ${coverCls ?? "bg-foreground/[0.06] text-foreground/70"}`}
        title={reason ?? undefined}
      >
        {inner}
      </span>
    );
  }

  return (
    <span className="relative inline-flex">
      <button
        ref={btnRef}
        type="button"
        onClick={() => (open ? setOpen(false) : openMenu())}
        aria-haspopup="menu"
        aria-expanded={open}
        title={reason ?? "Dataset type"}
        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 h-6 text-[11px] font-medium uppercase tracking-wider transition-colors ${coverCls ?? "bg-foreground/[0.06] hover:bg-foreground/[0.1] text-foreground/70"}`}
      >
        {inner}
        <svg
          viewBox="0 0 24 24"
          className={`h-3 w-3 ${onCover ? "opacity-60" : "text-foreground/40"}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && menuPos && typeof document !== "undefined" && createPortal(
        <>
          {/* Click-away backdrop. */}
          <div className="fixed inset-0 z-[100]" onClick={() => setOpen(false)} aria-hidden />
          <div
            role="menu"
            style={{ position: "fixed", top: menuPos.top, left: menuPos.left, width: MENU_WIDTH }}
            className="z-[101] rounded-xl border border-foreground/10 bg-[var(--background)] p-3 shadow-xl"
          >
            <p className="text-[10px] uppercase tracking-wider text-foreground/40">
              Dataset type
            </p>
            {reason && (
              <p className="mt-1.5 text-xs leading-relaxed text-foreground/65">{reason}</p>
            )}
            <div className="mt-3 flex flex-col gap-1">
              {OPTIONS.map((opt) => {
                const active =
                  (opt.key === "auto" && value.source === "auto") ||
                  (opt.key !== "auto" && value.source !== "auto" && value.type === opt.key);
                return (
                  <button
                    key={opt.key}
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setOpen(false);
                      onChange(opt.key);
                    }}
                    className={[
                      "w-full rounded-lg px-2.5 py-1.5 text-left transition-colors",
                      active
                        ? "bg-foreground/[0.08]"
                        : "hover:bg-foreground/[0.05]",
                    ].join(" ")}
                  >
                    <span className="flex items-center gap-2 text-sm text-foreground/90">
                      {opt.label}
                      {active && (
                        <svg
                          viewBox="0 0 24 24"
                          className="h-3.5 w-3.5 text-amber-400"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden
                        >
                          <path d="M20 6L9 17l-5-5" />
                        </svg>
                      )}
                    </span>
                    <span className="block text-[11px] text-foreground/45">{opt.desc}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </>,
        document.body,
      )}
    </span>
  );
}
