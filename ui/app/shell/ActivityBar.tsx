"use client";

// VS Code-style activity bar: a 48px vertical icon column on the left
// edge. Explorer / Guide at the top, Settings gear pinned to the
// bottom. The active item shows a 2px left accent and renders at
// full opacity; inactive items sit at 55% and lift to 85% on hover.
// (Models is invisible plumbing now - the engine downloads and loads
// everything itself, so there is no Models activity.)

import type { ReactNode } from "react";

export type ActivityKey = "explorer" | "guide";

const STROKE = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function ExplorerIcon() {
  // Two stacked file outlines (files/tree explorer).
  return (
    <svg viewBox="0 0 24 24" width={20} height={20} {...STROKE} aria-hidden>
      <path d="M15 3H8a1.5 1.5 0 0 0-1.5 1.5V17A1.5 1.5 0 0 0 8 18.5h9a1.5 1.5 0 0 0 1.5-1.5V6.5L15 3Z" />
      <path d="M15 3v3.5h3.5" />
      <path d="M4.5 7.5V19A1.5 1.5 0 0 0 6 20.5h9" />
    </svg>
  );
}

function GuideIcon() {
  // Open book.
  return (
    <svg viewBox="0 0 24 24" width={20} height={20} {...STROKE} aria-hidden>
      <path d="M12 6.5C10.5 5 8.5 4.5 5.5 4.5c-.6 0-1 .4-1 1v12c0 .6.4 1 1 1 3 0 5 .5 6.5 2 1.5-1.5 3.5-2 6.5-2 .6 0 1-.4 1-1v-12c0-.6-.4-1-1-1-3 0-5 .5-6.5 2Z" />
      <path d="M12 6.5v14" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" width={20} height={20} {...STROKE} aria-hidden>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.03 1.56V21a2 2 0 1 1-4 0v-.09A1.7 1.7 0 0 0 8.98 19.4a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1.03H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.6 8.98a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34H9a1.7 1.7 0 0 0 1.03-1.56V3a2 2 0 1 1 4 0v.09c0 .68.4 1.3 1.03 1.56a1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87V9c.26.63.88 1.03 1.56 1.03H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.56 1.03Z" />
    </svg>
  );
}

function ActivityButton({
  active,
  label,
  onClick,
  children,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={[
        "relative grid h-12 w-12 place-items-center text-[var(--foreground)] transition-opacity",
        active ? "opacity-100" : "opacity-55 hover:opacity-85",
      ].join(" ")}
    >
      {active && (
        <span
          aria-hidden
          className="absolute left-0 top-2 bottom-2 w-[2px] bg-[var(--accent)]"
        />
      )}
      {children}
    </button>
  );
}

export function ActivityBar({
  activity,
  onSelect,
  onSettings,
  settingsActive = false,
}: {
  activity: ActivityKey;
  /** Fired for the view icons. Explorer always navigates back to the
   *  workspace (and re-expands a collapsed side bar - collapsing lives
   *  on the Explorer pane header); Guide switches to the guide view. */
  onSelect: (key: ActivityKey) => void;
  onSettings: () => void;
  settingsActive?: boolean;
}) {
  return (
    <nav
      aria-label="Activity bar"
      className="flex w-12 shrink-0 flex-col items-center border-r border-[var(--border)] bg-[var(--background)]"
    >
      <ActivityButton
        active={activity === "explorer"}
        label="Explorer"
        onClick={() => onSelect("explorer")}
      >
        <ExplorerIcon />
      </ActivityButton>
      <ActivityButton
        active={activity === "guide"}
        label="Guide"
        onClick={() => onSelect("guide")}
      >
        <GuideIcon />
      </ActivityButton>
      <div className="flex-1" aria-hidden />
      <ActivityButton active={settingsActive} label="Settings" onClick={onSettings}>
        <GearIcon />
      </ActivityButton>
    </nav>
  );
}
