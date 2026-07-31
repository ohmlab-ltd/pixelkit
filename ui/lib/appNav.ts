// Tiny in-process event bus for switching tabs inside /app from
// arbitrary nested components (PlanPill, BillingWarning, UsagePanel,
// the VLM upgrade link in ProjectView, etc.) without prop-drilling
// or URL navigation. /app/page.tsx subscribes and updates its tab
// state in response.

export type AppTab = "workspaces" | "projects" | "guide" | "pricing" | "terminal";

const EVENT_NAME = "pixelkit:navigate";
const CURRENT_EVENT = "pixelkit:current-tab";

export function navigateAppTo(tab: AppTab) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<AppTab>(EVENT_NAME, { detail: tab }));
}

export function onAppNavigate(handler: (tab: AppTab) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const listener = (e: Event) => {
    const tab = (e as CustomEvent<AppTab>).detail;
    if (tab) handler(tab);
  };
  window.addEventListener(EVENT_NAME, listener);
  return () => window.removeEventListener(EVENT_NAME, listener);
}

// Broadcast the resolved current tab from /app/page.tsx. Components
// outside the page (e.g. ScrollToTop in the root layout) can listen
// to gate themselves on a specific tab without having to read the
// query string, the URL doesn't update when tabs switch in-place,
// so the query-string approach mis-fires on initial load + tab
// changes alike.
export function broadcastCurrentTab(tab: AppTab) {
  if (typeof window === "undefined") return;
  // Stash on the window so late-mounting listeners can read the
  // current state immediately instead of waiting for the next change.
  (window as unknown as { __pixelkitCurrentTab?: AppTab }).__pixelkitCurrentTab = tab;
  window.dispatchEvent(new CustomEvent<AppTab>(CURRENT_EVENT, { detail: tab }));
}

export function onCurrentTabChange(handler: (tab: AppTab | null) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const listener = (e: Event) => {
    const tab = (e as CustomEvent<AppTab>).detail;
    handler(tab ?? null);
  };
  window.addEventListener(CURRENT_EVENT, listener);
  return () => window.removeEventListener(CURRENT_EVENT, listener);
}

export function readCurrentTab(): AppTab | null {
  if (typeof window === "undefined") return null;
  return (window as unknown as { __pixelkitCurrentTab?: AppTab }).__pixelkitCurrentTab ?? null;
}
