"use client";

// Side bar hosting the Explorer pane.

import { ExplorerPane, type ExplorerDataset, type DatasetSection } from "./ExplorerPane";

export type { ExplorerDataset, DatasetSection };

export function SideBar({
  selectedDatasetId,
  activeSection,
  onOpenDataset,
  onOpenSection,
  onOpenProject,
  onNewDataset,
  onCollapse,
}: {
  selectedDatasetId: string | null;
  /** Highlights the matching third-level tree row. */
  activeSection: DatasetSection | null;
  onOpenDataset: (ds: ExplorerDataset) => void;
  /** Opens `ds` if needed and jumps to `section`. */
  onOpenSection: (ds: ExplorerDataset, section: DatasetSection) => void;
  /** Opens the Project page for a top-level tree row. */
  onOpenProject: (containerId: string) => void;
  onNewDataset: () => void;
  onCollapse: () => void;
}) {
  return (
    <aside
      aria-label="Explorer"
      className="w-[260px] shrink-0 border-r border-[var(--border)] bg-foreground/[0.02] min-h-0"
    >
      <ExplorerPane
        selectedDatasetId={selectedDatasetId}
        activeSection={activeSection}
        onOpenDataset={onOpenDataset}
        onOpenSection={onOpenSection}
        onOpenProject={onOpenProject}
        onNewDataset={onNewDataset}
        onCollapse={onCollapse}
      />
    </aside>
  );
}
