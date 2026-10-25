"use client";

import type { ReactNode } from "react";

// CSS-only hover tooltip above its children. The wrapping span takes the
// hover, so a disabled button still shows it.
export function Tooltip({
  label,
  children,
  width = 240,
  className = "",
}: {
  label: ReactNode;
  children: ReactNode;
  /** px */
  width?: number;
  className?: string;
}) {
  return (
    <span className={["relative inline-flex group", className].join(" ")}>
      {children}
      <span
        role="tooltip"
        className={[
          "pointer-events-none absolute z-50 select-none whitespace-normal",
          "bottom-full mb-2 left-1/2 -translate-x-1/2",
          "rounded-md border border-white/10 px-2.5 py-1.5",
          "text-[11px] leading-snug font-medium text-white/90",
          "opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0",
          "transition-[opacity,transform] duration-100 ease-out",
        ].join(" ")}
        style={{
          background: "rgba(20,20,22,0.96)",
          backdropFilter: "blur(8px)",
          WebkitBackdropFilter: "blur(8px)",
          boxShadow: "0 6px 16px -4px rgb(var(--shadow-rgb) / 0.6), 0 0 0 1px rgb(var(--foreground-rgb) / 0.02) inset",
          width: `${width}px`,
          maxWidth: "min(80vw, 320px)",
        }}
      >
        {label}
      </span>
    </span>
  );
}
