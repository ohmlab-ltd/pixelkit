"use client";

// Workspace Projects row: a New project tile and a card per Project.
import { useCallback, useEffect, useState } from "react";

import { CreateProjectModal } from "./CreateProjectModal";
import { containerCoverUrl, listContainers, type ContainerCard } from "@/lib/containers";
import { timeAgo } from "@/lib/timeAgo";

// The monogram shows for a Project without a cover, or once the retries fail.
function ContainerCover({
  id,
  name,
  cover,
  coverUpdated,
}: {
  id: string;
  name: string;
  cover: string | null;
  coverUpdated?: string;
}) {
  const [retry, setRetry] = useState(0);
  const [failed, setFailed] = useState(false);
  if (!cover || failed) {
    return (
      <div className="flex h-full w-full items-center justify-center text-3xl font-bold text-white/80 drop-shadow-sm transition-transform duration-300 group-hover:scale-110">
        {(name || "?").slice(0, 1).toUpperCase()}
      </div>
    );
  }
  // Retried with backoff, as the cover can 404 right after upload; ?r=N forces
  // a fresh request. Loaded eagerly.
  const base = containerCoverUrl(id, coverUpdated);
  const src = retry > 0 ? `${base}${base.includes("?") ? "&" : "?"}r=${retry}` : base;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      key={src}
      src={src}
      alt=""
      onError={() => {
        if (retry >= 3) { setFailed(true); return; }
        const next = retry + 1;
        window.setTimeout(() => setRetry(next), 400 * Math.pow(2, retry));
      }}
      className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]"
    />
  );
}

function ProjectCard({ card, onClick }: { card: ContainerCard; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="pk-card pk-card-hover group flex flex-col overflow-hidden rounded-2xl text-left"
    >
      <div className="pk-cover relative aspect-[16/9] w-full overflow-hidden">
        <ContainerCover
          key={card.cover_updated ?? "cover"}
          id={card.id}
          name={card.name}
          cover={card.cover}
          coverUpdated={card.cover_updated}
        />
      </div>
      <div className="flex flex-1 flex-col gap-2 p-3.5">
        <span className="flex items-center gap-2 truncate font-semibold text-foreground/90">
          <span className="truncate">{card.name}</span>
        </span>
        <span className="mt-auto flex items-center gap-2 pt-1 text-xs font-medium text-[var(--muted)]">
          <span className="inline-flex items-center gap-1">
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
              <rect x="3" y="4" width="18" height="6" rx="1.5" />
              <rect x="3" y="14" width="18" height="6" rx="1.5" />
            </svg>
            {card.n_datasets} dataset{card.n_datasets === 1 ? "" : "s"}
          </span>
          {card.updated && (
            <>
              <span aria-hidden className="text-foreground/25">·</span>
              <span>{timeAgo(card.updated)}</span>
            </>
          )}
        </span>
      </div>
    </button>
  );
}

function NewProjectTile({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex aspect-[16/9] flex-col items-center justify-center gap-2.5 rounded-2xl border-2 border-dashed border-foreground/15 text-[var(--muted)] transition-colors hover:border-orange-400/70 hover:bg-orange-500/[0.04] hover:text-orange-500 sm:aspect-auto sm:min-h-[148px]"
    >
      <span className="grid h-11 w-11 place-items-center rounded-2xl bg-foreground/[0.04] transition-colors group-hover:bg-orange-500/[0.12]">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
          <path d="M12 5v14M5 12h14" />
        </svg>
      </span>
      <span className="text-sm font-semibold">New project</span>
    </button>
  );
}

// Module-level cache so the row paints at once on return, then refreshes.
let _containersCache: ContainerCard[] | null = null;

export function ProjectsSection({ onOpenProject }: { onOpenProject?: (id: string) => void }) {
  const [cards, setCards] = useState<ContainerCard[] | null>(() => _containersCache);
  const [showCreate, setShowCreate] = useState(false);

  const reload = useCallback(() => {
    listContainers().then((c) => {
      // null: the fetch failed, so the last row stays. [] is a real empty list.
      if (c === null) return;
      _containersCache = c;
      setCards(c);
    });
  }, []);
  useEffect(() => {
    reload();
  }, [reload]);

  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-2.5">
        <span className="pk-accent-bar" aria-hidden />
        <h2 className="pk-section-title text-xl">Projects</h2>
        {cards && cards.length > 0 && (
          <span className="rounded-full bg-foreground/[0.06] px-2 py-0.5 text-xs font-semibold text-[var(--muted)]">
            {cards.length}
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        <NewProjectTile onClick={() => setShowCreate(true)} />
        {(cards ?? []).map((c) => (
          <ProjectCard
            key={c.id}
            card={c}
            onClick={() => onOpenProject?.(c.id)}
          />
        ))}
      </div>

      <CreateProjectModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={(c) => {
          setShowCreate(false);
          reload();
          onOpenProject?.(c.id);
        }}
      />
    </section>
  );
}
