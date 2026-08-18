// Tiny in-process event bus for switching tabs inside /app from
// arbitrary nested components without prop-drilling or URL
// navigation. /app/page.tsx subscribes and updates its tab state in
// response.

export type AppTab = "workspaces" | "projects" | "guide";

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

// "New dataset" request bus. The desktop shell's Explorer side bar
// exposes a "+" button, but the onboarding flow (name popup → labels →
// references) is owned entirely by HomeView. Rather than prop-drilling
// a begin-callback out of HomeView, the shell fires this event and
// HomeView (which stays mounted under the workspace view) reacts by
// opening its existing CreateDatasetModal - the exact same entry the
// "+ Add Dataset" toolbar button uses.
const NEW_DATASET_EVENT = "pixelkit:new-dataset";

export function requestNewDataset() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(NEW_DATASET_EVENT));
}

export function onNewDatasetRequest(handler: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const listener = () => handler();
  window.addEventListener(NEW_DATASET_EVENT, listener);
  return () => window.removeEventListener(NEW_DATASET_EVENT, listener);
}

// Explorer-refresh bus. The Explorer tree polls its listing every
// 10 s, which makes a freshly-created dataset invisible for up to a
// poll cycle. Mutation paths (create, delete, duplicate, rename,
// container add) fire this right after their API call resolves and
// the tree re-fetches immediately. Fire-and-forget: no payload, the
// listing endpoint is the source of truth.
const EXPLORER_REFRESH_EVENT = "pixelkit:explorer-refresh";

export function requestExplorerRefresh() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(EXPLORER_REFRESH_EVENT));
}

export function onExplorerRefresh(handler: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const listener = () => handler();
  window.addEventListener(EXPLORER_REFRESH_EVENT, listener);
  return () => window.removeEventListener(EXPLORER_REFRESH_EVENT, listener);
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
