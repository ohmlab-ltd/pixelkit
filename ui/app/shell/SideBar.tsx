"use client";

// Collapsible 260px side bar hosting the Explorer pane (the single
// workspace tree: Projects → datasets → dataset sections). The Models
// pane is gone — models are engine-managed plumbing with a passive
// status readout in Settings.

import { ExplorerPane, type ExplorerDataset, type DatasetSection } from "./ExplorerPane";

export type { ExplorerDataset, DatasetSection };

export function SideBar({
  username,
  selectedDatasetId,
  activeSection,
  onOpenDataset,
  onOpenSection,
  onNewDataset,
}: {
  username: string;
  selectedDatasetId: string | null;
  /** Active section of the open dataset — highlights the matching
   *  third-level tree row. */
  activeSection: DatasetSection | null;
  onOpenDataset: (ds: ExplorerDataset) => void;
  /** Open `ds` (if it isn't already open) and jump to `section`. */
  onOpenSection: (ds: ExplorerDataset, section: DatasetSection) => void;
  onNewDataset: () => void;
}) {
  return (
    <aside
      aria-label="Explorer"
      className="w-[260px] shrink-0 border-r border-[var(--border)] bg-foreground/[0.02] min-h-0"
    >
      <ExplorerPane
        username={username}
        selectedDatasetId={selectedDatasetId}
        activeSection={activeSection}
        onOpenDataset={onOpenDataset}
        onOpenSection={onOpenSection}
        onNewDataset={onNewDataset}
      />
    </aside>
  );
}
