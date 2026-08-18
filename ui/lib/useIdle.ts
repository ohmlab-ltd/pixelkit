"use client";

import { useEffect, useRef, useState } from "react";

// Returns true after `timeoutMs` of no user activity. "Activity" =
// mousemove / mousedown / keydown / touchstart / scroll / focus on
// the window. Resets the timer on each.
//
// Used to suspend background polling (augment job active, sidecar
// signal) so an idle tab doesn't keep firing /api/v2 requests
// every 2 s - eats Vercel observability quota for no user-visible
// benefit. Activity flips us back to active immediately so the
// next genuine interaction sees up-to-date state.
export function useIdle(timeoutMs: number = 90_000): boolean {
  const [idle, setIdle] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const reset = () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      // Optimistic: flip back to active the moment any event fires.
      setIdle((cur) => (cur ? false : cur));
      timerRef.current = window.setTimeout(() => {
        setIdle(true);
      }, timeoutMs);
    };

    // Bind a single listener per event type. `passive: true` keeps
    // scroll/touch off the main thread's blocking path.
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
    // Visibility change: when the tab becomes visible again,
    // treat as activity. When it goes hidden, force idle
    // immediately rather than waiting out the timer.
    const onVis = () => {
      if (document.visibilityState === "hidden") {
        if (timerRef.current !== null) window.clearTimeout(timerRef.current);
        setIdle(true);
      } else {
        reset();
      }
    };
    document.addEventListener("visibilitychange", onVis);

    // Seed: start the timer so first idle hit lands `timeoutMs`
    // after mount, even if the user never moves.
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
