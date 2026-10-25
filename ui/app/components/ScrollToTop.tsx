"use client";

// Floating "Back to top" button. It follows the nearest scrollable ancestor,
// or the window when there is none.

import { useEffect, useRef, useState } from "react";

// px of scroll before the button shows.
const THRESHOLD = 400;

function findScrollContainer(from: HTMLElement | null): HTMLElement | null {
  let el = from?.parentElement ?? null;
  while (el) {
    const overflowY = window.getComputedStyle(el).overflowY;
    if (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay") {
      return el;
    }
    el = el.parentElement;
  }
  return null;
}

export function ScrollToTop() {
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const containerRef = useRef<HTMLElement | null>(null);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const container = findScrollContainer(btnRef.current);
    containerRef.current = container;
    const target: HTMLElement | Window = container ?? window;
    const update = () =>
      setShow(
        (container ? container.scrollTop : window.scrollY) > THRESHOLD,
      );
    update();
    target.addEventListener("scroll", update, { passive: true });
    return () => target.removeEventListener("scroll", update);
  }, []);

  const scrollToTop = () => {
    const container = containerRef.current;
    if (container) container.scrollTo({ top: 0, behavior: "smooth" });
    else window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <button
      ref={btnRef}
      type="button"
      onClick={scrollToTop}
      aria-label="Scroll to top"
      tabIndex={show ? 0 : -1}
      className="fixed left-1/2 top-14 z-[1100] -translate-x-1/2 inline-flex items-center gap-2 rounded-md border border-foreground/15 px-4 py-2 text-[12px] uppercase tracking-[0.18em] font-mono text-[var(--foreground)] hover:bg-foreground/[0.05] transition-colors"
      style={{
        opacity: show ? 1 : 0,
        // top-14 clears the 36 px shell title bar.
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
