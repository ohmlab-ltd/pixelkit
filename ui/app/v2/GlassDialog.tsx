"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";

// One consistent modal/dialog surface for the whole project area: a blurred
// theme-aware glass panel over a blurred backdrop, with Escape-to-close,
// outside-click-to-close, focus trapping, body-scroll lock and an accessible
// close button. Styling comes from the shared `.pk-glass` / `.pk-backdrop`
// tokens in globals.css so it reads correctly in light and dark mode, and the
// `.pk-pop` entrance respects prefers-reduced-motion. Additive: existing
// modals adopt it incrementally without changing their logic.
export function GlassDialog({
  open,
  onClose,
  title,
  children,
  maxWidth = "max-w-lg",
  showClose = true,
}: {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  maxWidth?: string;
  showClose?: boolean;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);
  // Keep the latest onClose in a ref so the focus/key effect depends ONLY on
  // `open`. A parent passing an inline onClose recreates it every render; if the
  // effect depended on it, it would re-run on every keystroke and yank focus
  // back to the first control (the close button) -- making typing impossible.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    restoreFocus.current = (document.activeElement as HTMLElement) ?? null;
    const panel = panelRef.current;
    const focusables = (): HTMLElement[] =>
      panel
        ? Array.from(
            panel.querySelectorAll<HTMLElement>(
              'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])',
            ),
          ).filter((el) => el.offsetParent !== null)
        : [];
    (focusables()[0] ?? panel)?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key === "Tab" && panel) {
        const f = focusables();
        if (f.length === 0) {
          e.preventDefault();
          return;
        }
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = prevOverflow;
      restoreFocus.current?.focus?.();
    };
  }, [open]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="pk-backdrop fixed inset-0 z-[100] flex items-start justify-center overflow-y-auto p-4 pt-[8vh] sm:pt-[10vh]"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose(); // outside click
      }}
      role="presentation"
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
        className={`pk-glass pk-pop relative w-full ${maxWidth} rounded-xl outline-none`}
      >
        {(title || showClose) && (
          <div className="flex items-center justify-between gap-3 px-5 pb-3 pt-4">
            {title ? (
              <h2 className="text-[16px] font-semibold tracking-tight text-foreground">{title}</h2>
            ) : (
              <span />
            )}
            {showClose && (
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="grid h-8 w-8 shrink-0 place-items-center rounded-md text-foreground/55 outline-none transition-colors hover:bg-[var(--surface-hover)] hover:text-foreground focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]"
              >
                <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none">
                  <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
                </svg>
              </button>
            )}
          </div>
        )}
        <div className="px-5 pb-5">{children}</div>
      </div>
    </div>,
    document.body,
  );
}
