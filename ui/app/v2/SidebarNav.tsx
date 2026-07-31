"use client";

import type { ReactNode } from "react";

// Vertical project-section nav for the left sidebar. Subtle-but-clear active
// state using the orange accent + a soft background, with a line icon per
// section. Pure presentation: it calls onSelect(key); all tab/routing logic
// stays in the parent.
export type SidebarItem = { key: string; label: string; count?: number | null; disabled?: boolean; disabledHint?: string };

// Line icons keyed by section. 16x16, stroke-based so they inherit currentColor
// (active = orange, idle = muted) and stay crisp in both themes.
function NavIcon({ name }: { name: string }): ReactNode {
  const p = (d: string) => <path d={d} />;
  const common = {
    viewBox: "0 0 24 24",
    className: "h-4 w-4 shrink-0",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  switch (name) {
    case "overview":
      return <svg {...common}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>;
    case "references":
      return <svg {...common}><rect x="3" y="4" width="18" height="16" rx="2" />{p("M3 16l5-5 4 4 3-3 6 6")}<circle cx="9" cy="9" r="1.4" /></svg>;
    case "dataset":
      return <svg {...common}>{p("M4 7l8-4 8 4-8 4-8-4z")}{p("M4 12l8 4 8-4")}{p("M4 17l8 4 8-4")}</svg>;
    case "augmentations":
      return <svg {...common}>{p("M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18")}<circle cx="12" cy="12" r="2.5" /></svg>;
    case "annotations":
      return <svg {...common}>{p("M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0l-6-6a2 2 0 0 1 0-2.8l7.2-7.2a2 2 0 0 1 1.4-.6H19a2 2 0 0 1 2 2v5.4a2 2 0 0 1-.4 1z")}<circle cx="16.5" cy="7.5" r="1.2" /></svg>;
    case "train":
      return <svg {...common}><rect x="6" y="6" width="12" height="12" rx="2" />{p("M9 1.5v3M15 1.5v3M9 19.5v3M15 19.5v3M1.5 9h3M1.5 15h3M19.5 9h3M19.5 15h3")}</svg>;
    case "quantise":
      return <svg {...common}>{p("M9 4H6a2 2 0 0 0-2 2v3M15 4h3a2 2 0 0 1 2 2v3M20 15v3a2 2 0 0 1-2 2h-3M4 15v3a2 2 0 0 0 2 2h3M8 12h8")}</svg>;
    case "analyse":
      return <svg {...common}><circle cx="11" cy="11" r="7" />{p("M21 21l-4.3-4.3M8.5 11l2 2 3.5-4")}</svg>;
    case "deploy":
      return <svg {...common}>{p("M12 16V4M8 8l4-4 4 4M5 20h14")}</svg>;
    default:
      return <svg {...common}><circle cx="12" cy="12" r="8" /></svg>;
  }
}

export function SidebarNav({
  items,
  active,
  onSelect,
  framed = false,
}: {
  items: SidebarItem[];
  active: string;
  onSelect: (key: string) => void;
  // When true, wrap the items in a soft glass panel so the nav reads as one
  // cohesive unit (used for the floating gutter + pinned modes). The overlay
  // drawer is already a panel, so it passes framed=false.
  framed?: boolean;
}) {
  return (
    <nav
      className={[
        "flex flex-col gap-0.5",
        framed
          ? "rounded-2xl border border-[var(--modal-border)] bg-[var(--modal-surface)] p-1.5 shadow-md backdrop-blur-xl"
          : "",
      ].join(" ")}
      aria-label="Project sections"
    >
      {items.map((it) => {
        const on = active === it.key;
        const disabled = !!it.disabled;
        return (
          <button
            key={it.key}
            type="button"
            onClick={() => { if (!disabled) onSelect(it.key); }}
            disabled={disabled}
            aria-current={on ? "page" : undefined}
            title={disabled ? it.disabledHint : undefined}
            className={[
              "group flex items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13px] font-medium outline-none transition-colors",
              "focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]",
              disabled
                ? "cursor-not-allowed text-foreground/25"
                : on
                  ? "bg-[rgb(var(--accent-orange-rgb)/0.12)] text-[var(--accent-orange)]"
                  : "text-foreground/60 hover:bg-foreground/[0.05] hover:text-foreground/90",
            ].join(" ")}
          >
            <NavIcon name={it.key} />
            <span className="flex-1 truncate">{it.label}</span>
            {typeof it.count === "number" && !disabled && (
              <span className={`shrink-0 tabular-nums text-[11px] ${on ? "text-[var(--accent-orange)]/80" : "text-foreground/35"}`}>
                {it.count}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
