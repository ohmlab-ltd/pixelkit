// Window events between the shell and nested components.

// The side bar "+" fires this; HomeView opens its CreateDatasetModal.
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

// Fired after a mutation so the Explorer tree refetches before its 10 s poll.
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
