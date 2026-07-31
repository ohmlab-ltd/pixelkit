"use client";

import type { ReactNode } from "react";

// Vertical project-section nav for the dataset view's left column.
// Shell density: flat 13px rows (~26px tall) with a flat selected
// state, matching the Explorer pane. Pure presentation: it calls
// onSelect(key); all tab/routing logic stays in the parent.
export type SidebarItem = { key: string; label: string; count?: number | null; disabled?: boolean; disabledHint?: string };

// Line icons keyed by section, stroke-based so they inherit
// currentColor and stay crisp in both themes.
function NavIcon({ name }: { name: string }): ReactNode {
  const p = (d: string) => <path d={d} />;
  const common = {
    viewBox: "0 0 24 24",
    className: "h-3.5 w-3.5 shrink-0",
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
    default:
      return <svg {...common}><circle cx="12" cy="12" r="8" /></svg>;
  }
}

export function SidebarNav({
  items,
  active,
  onSelect,
}: {
  items: SidebarItem[];
  active: string;
  onSelect: (key: string) => void;
}) {
  return (
    <nav className="flex flex-col" aria-label="Project sections">
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
              "flex h-[26px] items-center gap-2 px-2.5 text-left text-[13px] outline-none transition-colors",
              "focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]",
              disabled
                ? "cursor-not-allowed text-foreground/25"
                : on
                  ? "bg-foreground/[0.08] text-[var(--foreground)]"
                  : "text-foreground/70 hover:bg-foreground/[0.05] hover:text-foreground/95",
            ].join(" ")}
          >
            <NavIcon name={it.key} />
            <span className="min-w-0 flex-1 truncate">{it.label}</span>
            {typeof it.count === "number" && !disabled && (
              <span className="shrink-0 tabular-nums text-[11px] text-foreground/35">
                {it.count}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
