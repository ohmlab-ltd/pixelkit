"use client";

// Collapsible 260px side bar. Which pane renders is driven by the
// active activity-bar item: Explorer (workspace tree) or Models.

import { ExplorerPane, type ExplorerDataset } from "./ExplorerPane";
import { ModelsPane } from "./ModelsPane";

export type { ExplorerDataset };

export function SideBar({
  pane,
  username,
  selectedDatasetId,
  onOpenDataset,
  onNewDataset,
}: {
  pane: "explorer" | "models";
  username: string;
  selectedDatasetId: string | null;
  onOpenDataset: (ds: ExplorerDataset) => void;
  onNewDataset: () => void;
}) {
  return (
    <aside
      aria-label={pane === "models" ? "Models" : "Explorer"}
      className="w-[260px] shrink-0 border-r border-[var(--border)] bg-foreground/[0.02] min-h-0"
    >
      {pane === "models" ? (
        <ModelsPane />
      ) : (
        <ExplorerPane
          username={username}
          selectedDatasetId={selectedDatasetId}
          onOpenDataset={onOpenDataset}
          onNewDataset={onNewDataset}
        />
      )}
    </aside>
  );
}
