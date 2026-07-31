"use client";

// Desktop-shell title bar (36px). Draggable so the packaged Electron
// window can be moved by grabbing it; interactive children must opt
// out with `no-drag` (none exist yet). On macOS under Electron the
// native traffic-light buttons overlay the top-left corner, so the
// app name shifts right by 76px only in that environment.

import { useEffect, useState, type CSSProperties } from "react";

// `-webkit-app-region` isn't in React's CSSProperties; a cast keeps
// the style typed without widening everything to `any`.
const DRAG: CSSProperties = { WebkitAppRegion: "drag" } as CSSProperties;

export function TitleBar({ title }: { title?: string }) {
  // Electron-on-macOS detection is a client-only concern; start false
  // so SSR/static export renders the no-inset variant and correct
  // padding applies after hydration.
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
      {/* Centre slot: the open dataset/project name. Absolutely centred
          so it doesn't drift when the left padding changes. */}
      {title ? (
        <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 max-w-[40%] truncate text-[13px] text-foreground/60 leading-none">
          {title}
        </span>
      ) : null}
      {/* Right slot: intentionally empty for now. */}
    </header>
  );
}
