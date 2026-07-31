"use client";

// Floating "back to top" affordance. Appears once the page has been
// scrolled past ~800 px and slides off again when the user is near
// the top.
//
// No route-gating here, the previous approach tried to read the
// active /app tab from a window-broadcast bus, which had races on
// first paint. Render this component ONLY from the place that knows
// it should be visible (e.g. inside the Projects tab branch in
// /app/page.tsx) and gate visibility via the `show` prop / parent
// conditional. Simpler, predictable.
//
// Positioning: pinned to the TOP-CENTRE of the viewport (not the
// bottom-right corner) so it sits where the user's eye lands after
// scrolling, with a blurred-glass backdrop so it reads clearly over
// busy gallery content underneath.

import { useEffect, useState } from "react";

// Trigger threshold in px. Set low enough that even a modest amount
// of scrolling reveals the button, earlier value of 800 meant a
// short workspace grid never crossed it on common laptop heights.
const THRESHOLD = 400;

export function ScrollToTop() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const update = () => setShow(window.scrollY > THRESHOLD);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);

  return (
    <button
      type="button"
      onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      aria-label="Scroll to top"
      tabIndex={show ? 0 : -1}
      className="fixed left-1/2 top-20 z-[1100] -translate-x-1/2 inline-flex items-center gap-2 rounded-full border border-foreground/15 px-5 py-2.5 text-[12px] uppercase tracking-[0.18em] font-mono text-[var(--foreground)] hover:bg-foreground/[0.05] transition-colors"
      style={{
        opacity: show ? 1 : 0,
        // TopNav is sticky at top:12 px with h-14 (= 68 px bottom edge)
        // so we anchor below that. Reveal animation slides DOWN from
        // the nav's underside.
        transform: `translateX(-50%) translateY(${show ? "0" : "-8px"})`,
        pointerEvents: show ? "auto" : "none",
        transition: "opacity 220ms ease, transform 220ms ease",
        background: "rgb(var(--surface-rgb) / 0.7)",
        backdropFilter: "blur(14px) saturate(120%)",
        WebkitBackdropFilter: "blur(14px) saturate(120%)",
        boxShadow: "var(--shadow-strong)",
      }}
    >
      <svg
        viewBox="0 0 24 24"
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <path d="M12 19V5" />
        <path d="M5 12l7-7 7 7" />
      </svg>
      <span>Back to top</span>
    </button>
  );
}
