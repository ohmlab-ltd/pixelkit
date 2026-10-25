"use client";

import { useEffect, useRef, useState } from "react";

// True after timeoutMs without window input, and while the tab is hidden.
export function useIdle(timeoutMs: number = 90_000): boolean {
  const [idle, setIdle] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const reset = () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      setIdle((cur) => (cur ? false : cur));
      timerRef.current = window.setTimeout(() => {
        setIdle(true);
      }, timeoutMs);
    };

    const opts: AddEventListenerOptions = { passive: true };
    const events: (keyof WindowEventMap)[] = [
      "mousemove",
      "mousedown",
      "keydown",
      "touchstart",
      "scroll",
      "focus",
    ];
    for (const e of events) {
      window.addEventListener(e, reset, opts);
    }
    const onVis = () => {
      if (document.visibilityState === "hidden") {
        if (timerRef.current !== null) window.clearTimeout(timerRef.current);
        setIdle(true);
      } else {
        reset();
      }
    };
    document.addEventListener("visibilitychange", onVis);

    reset();

    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      for (const e of events) {
        window.removeEventListener(e, reset, opts);
      }
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [timeoutMs]);

  return idle;
}
