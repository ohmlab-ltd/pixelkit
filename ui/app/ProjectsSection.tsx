"use client";

// Workspace "Projects" row: a New-Project tile + cards for each Project the
// user belongs to. A Project (container) holds many Datasets; this sits above
// the Datasets grid on the workspace. Cover/owner/last-updated/#datasets per
// the spec. Clicking a card opens the Project page (onOpenProject).
import { useCallback, useEffect, useState } from "react";

import { CreateProjectModal } from "./CreateProjectModal";
import { Avatar } from "./components/Avatar";
import { containerCoverUrl, fetchAvatars, listContainers, type ContainerCard } from "@/lib/containers";

function timeAgo(iso?: string): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// Inline padlock shown to the right of a private Project's name — matches the
// amber PrivateLockIcon the dataset cards use, so Projects and datasets read the
// same way on the workspace.
function Lock() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-amber-600 dark:text-amber-300/80"
      aria-label="Private project"
      role="img"
    >
      <title>Private project</title>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </svg>
  );
}

// Always try to load the Project cover and fall back to the monogram only on a
// real load error. We don't gate on `card.cover` because that field has been
// arriving null even when a cover.jpg exists in R2; attempting the image
// directly is robust to that (and projects are few per user, so the occasional
// 404 for a cover-less project is cheap).
function ContainerCover({ id, name }: { id: string; name: string }) {
  const [retry, setRetry] = useState(0);
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-full w-full items-center justify-center text-3xl font-bold text-white/80 drop-shadow-sm transition-transform duration-300 group-hover:scale-110">
        {(name || "?").slice(0, 1).toUpperCase()}
      </div>
    );
  }
  // Load eagerly (projects are few) + a short backed-off retry. The cover GET
  // can transiently 404/5xx right after upload while R2/CDN propagates; without
  // a retry the card stuck on the monogram until it remounted — the "cover only
  // appears after I click into the project and back out" bug. The cache-buster
  // (?r=N) forces the retry to actually re-request rather than reuse the failed
  // response. No loading="lazy": deferral was part of why it didn't paint until
  // a remount put it in view.
  const base = containerCoverUrl(id);
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

function ProjectCard({ card, onClick, avatarUrl }: { card: ContainerCard; onClick: () => void; avatarUrl?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="pk-card pk-card-hover group flex flex-col overflow-hidden rounded-2xl text-left"
    >
      <div className="pk-cover relative aspect-[16/9] w-full overflow-hidden">
        <ContainerCover id={card.id} name={card.name} />
      </div>
      <div className="flex flex-1 flex-col gap-2 p-3.5">
        <span className="flex items-center gap-2 truncate font-semibold text-foreground/90">
          <span className="truncate">{card.name}</span>
          {card.private && <Lock />}
        </span>
        <span className="flex items-center gap-1.5 truncate text-xs text-[var(--muted)]">
          <Avatar name={card.owner} src={avatarUrl} className="h-4 w-4 shrink-0 rounded-full text-[8px] font-bold" />
          {card.owner}
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

// Module-level cache of the Projects row so navigating into a Project and back
// paints the cards instantly from memory (then refreshes), instead of flashing
// empty for the ~0.5s the listContainers fetch takes. Mirrors how the dataset
// list is cached.
let _containersCache: ContainerCard[] | null = null;
let _avatarsCache: Record<string, string> = {};

export function ProjectsSection({ onOpenProject }: { onOpenProject?: (id: string) => void }) {
  const [cards, setCards] = useState<ContainerCard[] | null>(() => _containersCache);
  const [showCreate, setShowCreate] = useState(false);
  const [avatars, setAvatars] = useState<Record<string, string>>(() => _avatarsCache);

  const reload = useCallback(() => {
    listContainers().then((c) => {
      // null = the fetch failed (expired bearer / network blip). Keep the
      // last-known row rather than wiping it to an empty "no projects" state;
      // apiFetch already retried once with a fresh token, so a null here means
      // it's genuinely unreachable for now. A real empty list is [] and DOES
      // update (so a deleted-last-project reflects correctly).
      if (c === null) return;
      _containersCache = c;
      setCards(c);
    });
  }, []);
  useEffect(() => {
    reload();
  }, [reload]);
  // Resolve the owners' real profile pictures (the backend only knows usernames).
  useEffect(() => {
    if (!cards || cards.length === 0) return;
    fetchAvatars(cards.map((c) => c.owner)).then((a) => {
      _avatarsCache = a;
      setAvatars(a);
    });
  }, [cards]);

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
            avatarUrl={avatars[(c.owner || "").toLowerCase()]}
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
