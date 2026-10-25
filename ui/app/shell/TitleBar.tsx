"use client";

// Shell title bar and Electron drag region. Interactive children need
// `no-drag`. On macOS the name clears the window buttons by 76 px.

import { useEffect, useState, type CSSProperties } from "react";

// `-webkit-app-region` is not in React's CSSProperties.
const DRAG: CSSProperties = { WebkitAppRegion: "drag" } as CSSProperties;

export function TitleBar({ title }: { title?: string }) {
  // Detected after hydration; static rendering has no inset.
  const [macElectron, setMacElectron] = useState(false);
  useEffect(() => {
    if (typeof navigator === "undefined") return;
    const ua = navigator.userAgent || "";
    const isElectron = ua.includes("Electron");
    const isMac =
      /Mac/i.test(navigator.platform || "") || ua.includes("Macintosh");
    setMacElectron(isElectron && isMac);
  }, []);

  return (
    <header
      style={DRAG}
      className="relative h-9 shrink-0 flex items-center border-b border-[var(--border)] bg-[var(--background)] select-none"
    >
      <span
        className="text-[13px] font-medium text-[var(--foreground)] leading-none"
        style={{ paddingLeft: macElectron ? 76 : 12 }}
      >
        PixelKit
      </span>
      {title ? (
        <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 max-w-[40%] truncate font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--fg-dim)] leading-none">
          {title}
        </span>
      ) : null}
    </header>
  );
}
