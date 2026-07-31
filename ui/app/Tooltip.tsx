"use client";

import type { ReactNode } from "react";

// Instant CSS-only tooltip matching the floating-capsule aesthetic
// used elsewhere in the app: zinc-glass fill, white/10 border, mono
// uppercase tracking. Wraps an arbitrary child so disabled buttons
// can still trigger the tooltip, disabled <button>s suppress mouse
// events, so we hover the wrapping span instead and use group-hover.
//
// Pure :hover via Tailwind's `group` utility means there's no
// hydration delay and no setTimeout, appears the moment the cursor
// crosses the trigger.
export function Tooltip({
  label,
  children,
  side = "top",
  align = "center",
  variant = "label",
  width,
  className = "",
}: {
  label: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom";
  align?: "start" | "center" | "end";
  /** "label", small uppercase mono single-line. Use for chip / icon
   *  triggers where a short word is enough.
   *  "rich", sentence-case, normal weight, wraps to multi-line.
   *  Use for explanatory popups (e.g. ref-warning, dataset health).
   */
  variant?: "label" | "rich";
  /** Optional fixed pixel width when variant="rich", drives line
   *  wrapping. Defaults to 240 px. */
  width?: number;
  className?: string;
}) {
  if (!label) return <>{children}</>;

  const verticalPos =
    side === "top"
      ? "bottom-full mb-2"
      : "top-full mt-2";
  const horizontalPos =
    align === "start"
      ? "left-0"
      : align === "end"
      ? "right-0"
      : "left-1/2 -translate-x-1/2";

  const isRich = variant === "rich";
  const widthStyle: React.CSSProperties = isRich
    ? { width: `${width ?? 240}px`, maxWidth: "min(80vw, 320px)" }
    : {};
  const textClasses = isRich
    ? "text-[11px] leading-snug font-medium text-white/90"
    : "text-[10px] uppercase tracking-[0.18em] font-mono text-white/90";

  return (
    <span className={["relative inline-flex group", className].join(" ")}>
      {children}
      <span
        role="tooltip"
        className={[
          "pointer-events-none absolute z-50 select-none",
          isRich ? "whitespace-normal" : "whitespace-nowrap",
          verticalPos,
          horizontalPos,
          "rounded-md border border-white/10 px-2.5 py-1.5",
          textClasses,
          "opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0",
          "transition-[opacity,transform] duration-100 ease-out",
        ].join(" ")}
        style={{
          background: "rgba(20,20,22,0.96)",
          backdropFilter: "blur(8px)",
          WebkitBackdropFilter: "blur(8px)",
          boxShadow: "0 6px 16px -4px rgb(var(--shadow-rgb) / 0.6), 0 0 0 1px rgb(var(--foreground-rgb) / 0.02) inset",
          ...widthStyle,
        }}
      >
        {label}
      </span>
    </span>
  );
}
