"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { BlurhashCanvas } from "react-blurhash";
import { Footer } from "./Footer";
import { ProjectTagsRow } from "./components/ProjectTagsRow";
import type { ProjectSummary } from "./HomeView";
import { PrivateLockIcon, DerivedBadge, projectStatusBadges } from "./HomeView";
import { lookupUsers, type OwnerInfo as CachedOwnerInfo } from "@/lib/userCache";
import { listPublicContainers, containerCoverUrl, type ContainerCard } from "@/lib/containers";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SORTS = ["Trending", "Newest", "Most liked"] as const;
type Sort = (typeof SORTS)[number];

// FE → backend `sort` query value. Mirrors the modes the list
// endpoint supports so pagination stays stable across pages.
function sortToParam(s: Sort): string {
  if (s === "Newest") return "newest";
  if (s === "Most liked") return "most_liked";
  return "trending";
}

// Stable hash → hue, so each project gets a consistent colour without needing
// the backend to pick one.
function hueFor(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return Math.abs(h) % 360;
}

// Project (container) cover for the Community carousel: eager image with a
// monogram fallback on load error, mirroring the workspace Project card.
function PublicProjectCover({ id, name }: { id: string; name: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-foreground/[0.04] text-2xl font-bold text-white/80">
        {(name || "?").slice(0, 1).toUpperCase()}
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={containerCoverUrl(id)}
      alt=""
      onError={() => setFailed(true)}
      className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]"
    />
  );
}

type OwnerInfo = CachedOwnerInfo;

// localStorage cache for the first page of public projects. Hydrated
// on mount so the page paints instantly on repeat visits, total
// count + first 12 cards (with their blurhashes) are already there
// before any network fetch resolves. Cache invalidates after
// PUBLIC_PROJECTS_CACHE_TTL_MS or when the username changes (different
// user sees different filtered results).
const PUBLIC_PROJECTS_CACHE_KEY = "pixelkit_public_projects_v1";
const PUBLIC_PROJECTS_CACHE_TTL_MS = 5 * 60 * 1000;

type PublicProjectsCache = {
  username: string;
  total: number;
  items: ProjectSummary[];
  fetchedAt: number;
};

function readPublicProjectsCache(username: string): PublicProjectsCache | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(PUBLIC_PROJECTS_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PublicProjectsCache;
    if (parsed.username !== username) return null;
    if (Date.now() - (parsed.fetchedAt ?? 0) > PUBLIC_PROJECTS_CACHE_TTL_MS) return null;
    if (!Array.isArray(parsed.items)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writePublicProjectsCache(cache: PublicProjectsCache): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PUBLIC_PROJECTS_CACHE_KEY, JSON.stringify(cache));
  } catch {
    /* quota exceeded or other localStorage issue, non-fatal */
  }
}

export function ProjectsView({
  onOpen,
  username,
  loggedIn = true,
}: {
  onOpen: (id: string, owner?: string, displayName?: string, v2?: boolean) => void;
  username: string;
  loggedIn?: boolean;
}) {
  // Seed from the localStorage cache so the first paint has data ,
  // total count, first 12 cards including their blurhashes, without
  // waiting for the network round-trip. The fresh fetch below
  // overwrites these once it resolves so stale entries don't linger.
  const cached = typeof window !== "undefined" ? readPublicProjectsCache(username) : null;
  const [projects, setProjects] = useState<ProjectSummary[] | null>(cached?.items ?? null);
  const [total, setTotal] = useState<number | null>(cached?.total ?? null);
  const [owners, setOwners] = useState<Record<string, OwnerInfo>>({});
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  // Debounced search input, drives the actual /api/projects fetch so
  // we don't fire a request on every keystroke. Server-side `q` covers
  // every public project, not just the loaded slice.
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [sort, setSort] = useState<Sort>("Trending");
  // Infinite-scroll state. PAGE_SIZE matches the backend's typical
  // first-card paint window; we render this many fully on initial
  // load, then top up as the user scrolls. `loadingMore` debounces
  // the IntersectionObserver so a bottom-of-list trigger doesn't
  // fire 4 fetches before any of them resolve.
  const PAGE_SIZE = 12;
  const [loadingMore, setLoadingMore] = useState(false);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  // Spinner state for the filter row, true while the active-sort
  // refetch is in flight so the UI can paint a small spinning ring
  // next to the pills. Skips the initial cache-hydrated first paint
  // (handled via the initial-mount ref) so the spinner only appears
  // when the user explicitly switches sort.
  const [sortLoading, setSortLoading] = useState(false);
  // Public Projects (team containers) for the carousel above the dataset grid.
  const [publicProjects, setPublicProjects] = useState<ContainerCard[]>([]);
  useEffect(() => {
    let cancelled = false;
    listPublicContainers().then((c) => { if (!cancelled) setPublicProjects(c); });
    return () => { cancelled = true; };
  }, []);

  // Initial fetch, paginated. The backend returns
  // {total, items, offset, limit} when `limit` is in the query
  // string; we use `total` to size the placeholder grid up-front
  // so the user sees how many cards are coming before they all
  // resolve. Always runs even when we hydrated from cache so the
  // displayed data refreshes within ~200 ms of mount.
  // Debounce typing, coalesces fast input so we don't spam /api/projects.
  useEffect(() => {
    const t = window.setTimeout(() => setDebouncedQuery(query.trim()), 200);
    return () => window.clearTimeout(t);
  }, [query]);

  useEffect(() => {
    let cancelled = false;
    setSortLoading(true);
    (async () => {
      try {
        const qParam = debouncedQuery ? `&q=${encodeURIComponent(debouncedQuery)}` : "";
        const url = `${API}/api/projects?viewer=${encodeURIComponent(username)}&offset=0&limit=${PAGE_SIZE}&sort=${sortToParam(sort)}${qParam}`;
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error(`http ${r.status}`);
        const data = await r.json() as { total: number; items: ProjectSummary[] };
        if (cancelled) return;
        const items = data.items ?? [];
        const totalCount = data.total ?? items.length ?? 0;
        setProjects(items);
        setTotal(totalCount);
        // Only write-through the cache when not searching, the
        // filtered subset isn't representative of the "all public
        // projects" view we want to hydrate from next mount.
        if (!debouncedQuery) {
          writePublicProjectsCache({
            username,
            total: totalCount,
            items,
            fetchedAt: Date.now(),
          });
        }
        const usernames = Array.from(
          new Set(items.map((p) => p.createdBy).filter(Boolean)),
        );
        if (usernames.length > 0) {
          setOwners(await lookupUsers(usernames));
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setSortLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // Refetch on sort or query change so the backend re-orders +
    // paginates under the new mode. Cheap thanks to the response cache.
  }, [username, sort, debouncedQuery]);

  // Fetch the next page when the sentinel scrolls into view. Debounced
  // by `loadingMore` so a single trigger doesn't queue multiple fetches.
  useEffect(() => {
    if (projects === null || total === null) return;
    if (projects.length >= total) return;
    const node = sentinelRef.current;
    if (!node) return;
    const obs = new IntersectionObserver((entries) => {
      const hit = entries.some((e) => e.isIntersecting);
      if (!hit || loadingMore) return;
      setLoadingMore(true);
      (async () => {
        try {
          const qParam = debouncedQuery ? `&q=${encodeURIComponent(debouncedQuery)}` : "";
          const url = `${API}/api/projects?viewer=${encodeURIComponent(username)}&offset=${projects.length}&limit=${PAGE_SIZE}&sort=${sortToParam(sort)}${qParam}`;
          const r = await fetch(url, { cache: "no-store" });
          if (!r.ok) throw new Error(`http ${r.status}`);
          const data = await r.json() as { total: number; items: ProjectSummary[] };
          // Dedup by id before appending. Offset pagination can return a
          // project that's already loaded when the underlying list shifts
          // between page fetches — a delete (which removes a card and
          // shrinks the list), a "Trending" re-rank, or a like/favourite
          // reorder. Without this, the overlapping item is appended twice
          // and React renders two identical cards under the same key.
          setProjects((prev) => {
            const seen = new Set((prev ?? []).map((p) => p.id));
            const fresh = (data.items ?? []).filter((p) => !seen.has(p.id));
            return [...(prev ?? []), ...fresh];
          });
          setTotal(data.total ?? total);
          const newOwners = Array.from(
            new Set((data.items ?? []).map((p) => p.createdBy).filter(Boolean)),
          ).filter((u) => !(u in owners));
          if (newOwners.length > 0) {
            const more = await lookupUsers(newOwners);
            setOwners((prev) => ({ ...prev, ...more }));
          }
        } catch (e) {
          console.warn("[projects] paginate failed:", e);
        } finally {
          setLoadingMore(false);
        }
      })();
    }, { rootMargin: "300px" });
    obs.observe(node);
    return () => obs.disconnect();
  }, [projects, total, username, loadingMore, owners, sort, debouncedQuery]);

  // Patch any tag rename instantly when the project view dispatches
  // a meta-changed event, same hook as the workspace, so a label
  // rename inside the project shows up on the public card without
  // waiting for the next list refetch.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        projectId?: string;
        tags?: string[];
        label_aliases?: Record<string, string>;
        labelColours?: Record<string, string>;
        name?: string;
        private?: boolean;
        cover?: string | null;
      } | null | undefined;
      if (!detail?.projectId) return;
      setProjects((prev) =>
        prev?.map((p) =>
          p.id === detail.projectId
            ? {
                ...p,
                tags: detail.tags ?? p.tags,
                label_aliases: detail.label_aliases ?? p.label_aliases,
                labelColours: detail.labelColours ?? p.labelColours,
                name: detail.name ?? p.name,
                private: typeof detail.private === "boolean" ? detail.private : p.private,
              }
            : p,
        ) ?? null,
      );
    };
    const deletedHandler = (e: Event) => {
      const detail = (e as CustomEvent).detail as { projectId?: string } | null | undefined;
      if (!detail?.projectId) return;
      setProjects((prev) => prev?.filter((p) => p.id !== detail.projectId) ?? null);
    };
    window.addEventListener("pixelkit-project-meta-changed", handler);
    window.addEventListener("pixelkit-project-deleted", deletedHandler);
    return () => {
      window.removeEventListener("pixelkit-project-meta-changed", handler);
      window.removeEventListener("pixelkit-project-deleted", deletedHandler);
    };
  }, []);

  const toggleLike = async (project_id: string) => {
    if (!loggedIn || !username) return;
    setProjects((prev) =>
      prev?.map((p) =>
        p.id === project_id
          ? { ...p, likedByMe: !p.likedByMe, likes: p.likes + (p.likedByMe ? -1 : 1) }
          : p,
      ) ?? null,
    );
    try {
      const r = await fetch(`${API}/api/projects/${project_id}/like`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: username }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data: { likes: number; likedByMe: boolean } = await r.json();
      setProjects((prev) =>
        prev?.map((p) => (p.id === project_id ? { ...p, likes: data.likes, likedByMe: data.likedByMe } : p)) ?? null,
      );
    } catch {
      // ignore, next mount will sync
    }
  };

  const toggleFavourite = async (project_id: string) => {
    if (!loggedIn || !username) return;
    setProjects((prev) =>
      prev?.map((p) =>
        p.id === project_id
          ? { ...p, favouritedByMe: !p.favouritedByMe }
          : p,
      ) ?? null,
    );
    try {
      const r = await fetch(`${API}/api/projects/${project_id}/favourite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: username }),
      });
      if (!r.ok) throw new Error(`http ${r.status}`);
      const data: { favourites: number; favouritedByMe: boolean } = await r.json();
      setProjects((prev) =>
        prev?.map((p) =>
          p.id === project_id ? { ...p, favouritedByMe: data.favouritedByMe } : p,
        ) ?? null,
      );
    } catch {
      // ignore
    }
  };

  const list = useMemo(() => {
    // Filtering is server-side via the `q` param, the projects state
    // is already the matching slice. Order comes from the backend
    // (sort + favourites pinning), so we don't re-sort here.
    return projects ?? [];
  }, [projects]);

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <section className="mx-auto max-w-6xl px-6 pt-16 pb-10">
        <h1 className="text-5xl md:text-6xl font-medium tracking-tight">Community</h1>
        <p className="mt-4 max-w-xl text-foreground/50 text-lg">
          Browse community datasets and see what others are building.
        </p>
      </section>

      {/* Public Projects carousel: a horizontal rail of every public Project
          (team container), above the dataset grid. Clicking one opens the
          Project page. Hidden when there are no public Projects. */}
      {publicProjects.length > 0 && (
        <section className="mx-auto max-w-6xl px-6 pb-2">
          <div className="mb-3 flex items-center gap-2.5">
            <span className="pk-accent-bar" aria-hidden />
            <h2 className="text-lg font-medium tracking-tight">Public projects</h2>
            <span className="rounded-full bg-foreground/[0.06] px-2 py-0.5 text-xs font-semibold text-[var(--muted)]">
              {publicProjects.length}
            </span>
          </div>
          <div className="flex gap-4 overflow-x-auto pb-3 snap-x" style={{ scrollbarWidth: "thin" }}>
            {publicProjects.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => {
                  // Logged-in: open the in-app read-only Project page (it only
                  // mounts when authed). Logged-out: the standalone public page
                  // is the working fallback.
                  window.location.href = loggedIn ? `/app?project=${c.id}` : `/projects/${c.id}`;
                }}
                className="pk-card pk-card-hover group w-60 shrink-0 snap-start overflow-hidden rounded-2xl text-left"
              >
                <div className="pk-cover relative aspect-[16/9] w-full overflow-hidden">
                  <PublicProjectCover id={c.id} name={c.name} />
                </div>
                <div className="flex flex-col gap-1 p-3">
                  <span className="truncate font-semibold text-foreground/90">{c.name}</span>
                  <span className="truncate text-xs text-[var(--muted)]">
                    {c.n_datasets} dataset{c.n_datasets === 1 ? "" : "s"} · by {c.owner}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="mx-auto max-w-6xl px-6 pb-3 flex items-center gap-2 flex-wrap">
        {SORTS.map((s) => (
          <button
            key={s}
            onClick={() => setSort(s)}
            // Rounded tab chips, matching the workspace toolbar so the
            // two listing pages read as one family. Active pill snaps
            // (no ease); inactive pills carry a short hover tint.
            className={[
              "inline-flex items-center rounded-full px-3.5 h-9 text-sm transition-colors duration-[90ms]",
              sort === s
                ? "bg-foreground/[0.08] text-[var(--foreground)] font-medium"
                : "text-foreground/55 hover:text-foreground/90 hover:bg-foreground/[0.04]",
            ].join(" ")}
          >
            {s}
          </button>
        ))}
        {/* Small spinning ring next to the filter pills while the
            sort refetch is in flight. Fades in / out so the badge
            doesn't pop on fast cached hits. */}
        <span
          aria-hidden={!sortLoading}
          aria-label={sortLoading ? "Loading sorted projects" : undefined}
          className="inline-grid place-items-center h-5 w-5"
          style={{
            opacity: sortLoading ? 1 : 0,
            transition: "opacity 160ms ease-out",
          }}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-[var(--muted)] animate-spin" fill="none" aria-hidden>
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.22" strokeWidth="2.6" />
            <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
          </svg>
        </span>
        <span className="ml-auto text-xs text-[var(--muted)]">
          {/* Show the backend total when known and no search filter is
              active, without this the count rose 12 → 24 → ... as
              pagination batches landed, even though the feed had
              hundreds of projects on the server. */}
          {projects === null
            ? "loading…"
            : query.trim()
              ? `${total ?? list.length} match${(total ?? list.length) === 1 ? "" : "es"}`
              : `${total ?? list.length} project${(total ?? list.length) === 1 ? "" : "s"}`}
        </span>
        <label className="relative w-full sm:w-72">
          <span className="sr-only">Search projects</span>
          <svg
            aria-hidden
            viewBox="0 0 24 24"
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/40"
            fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search name, tag, author"
            className="h-9 w-full rounded-full bg-foreground/[0.04] border border-foreground/10 hover:border-foreground/20 focus:border-foreground/40 focus:bg-foreground/[0.06] outline-none pl-9 pr-3 text-sm text-[var(--foreground)] placeholder:text-foreground/40 transition-colors"
          />
        </label>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-24">
        {error && (
          <div className="rounded-xl border border-red-500/40 bg-red-500/[0.04] py-6 px-4 text-center text-sm text-red-300">
            Couldn&apos;t load projects: {error}
          </div>
        )}
        {!error && projects !== null && list.length === 0 && (
          <div className="rounded-3xl border border-foreground/10 bg-foreground/[0.02] py-16 text-center text-foreground/50">
            {query
              ? <>Nothing matches <span className="text-foreground/90">{query}</span>.</>
              : "No projects yet."}
          </div>
        )}
        {!error && list.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {list.map((p) => (
              <PublicCard
                key={p.id}
                project={p}
                owner={p.createdBy ? owners[p.createdBy] ?? null : null}
                loggedIn={loggedIn}
                onOpen={() => onOpen(p.id, p.createdBy, p.name, !!p.v2)}
                onLike={() => toggleLike(p.id)}
                onFavourite={() => toggleFavourite(p.id)}
              />
            ))}
            {/* Skeleton placeholders for projects that haven't been
                paginated in yet. Sized so the grid lands its final
                row count immediately, the user sees the full feed
                shape on first paint and tiles fill in as they
                scroll. Renders during search too, since `total`
                reflects the matching count. */}
            {total !== null
              && projects !== null
              && projects.length < total
              && Array.from({ length: Math.min(PAGE_SIZE, total - projects.length) }).map((_, i) => (
                <div
                  key={`skel-${projects.length + i}`}
                  className="rounded-2xl border border-foreground/10 bg-foreground/[0.02] overflow-hidden"
                >
                  <div className="aspect-video bg-foreground/[0.04]" />
                  <div className="p-4 space-y-2">
                    <div className="h-4 w-2/3 rounded bg-foreground/[0.06]" />
                    <div className="h-3 w-1/3 rounded bg-foreground/[0.04]" />
                  </div>
                </div>
              ))}
          </div>
        )}
        {/* Sentinel for the IntersectionObserver, sits 300 px
            above the bottom of the page so the next page kicks off
            before the user reaches the actual end. */}
        {total !== null && projects !== null && projects.length < total && (
          <div ref={sentinelRef} aria-hidden className="h-1 w-full" />
        )}
        {/* "Showing N of M" footer, gives the user instant feedback
            on the total project count even before all pages have
            loaded. */}
        {total !== null && total > 0 && !error && (
          <div className="mt-6 text-center text-[11px] text-foreground/35 font-mono">
            Showing {Math.min(list.length, total)} of {total}
            {loadingMore && " · loading…"}
          </div>
        )}
      </section>

      <Footer />
    </main>
  );
}

// Cover image with BlurHash placeholder + crossfade to the real
// image once it loads. The placeholder is decoded synchronously from
// `blurhash` (a ~30-char string from the backend list response) so
// the card has SOMETHING to show on the very first paint, even
// before any network round-trip. The full image fades in on top
// once <img> fires onLoad.
//
// V2 projects store covers under references/, V1 under imports/
// with an optional annotated preview overlay. Branch the URLs on
// the `v2` flag.
function ProjectThumbnail({
  projectId,
  thumbnail,
  updatedAt,
  alt,
  v2 = false,
  blurhash = null,
}: {
  projectId: string;
  thumbnail: string;
  updatedAt: string | null;
  alt: string;
  v2?: boolean;
  blurhash?: string | null;
}) {
  const stem = thumbnail.replace(/\.[^.]+$/, "");
  const cacheBuster = updatedAt ? `?v=${encodeURIComponent(updatedAt)}` : "";

  // /cover_thumb returns a ~480 px JPEG cached on disk, a fraction
  // of the bytes of the full-resolution cover. Works for both V1 and
  // V2 (the backend resolves the source from references/ then
  // imports/). Falls back to the unresized original if the thumb
  // endpoint 404s, e.g. on older deployments.
  const primaryUrl = `${API}/api/projects/${projectId}/cover_thumb`;
  const fallbackUrl = v2
    ? `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(thumbnail)}`
    : `${API}/api/projects/${projectId}/files/${encodeURIComponent(stem)}_annotated.jpg`;
  const finalFallbackUrl = v2
    ? `${API}/api/v2/projects/${projectId}/references/${encodeURIComponent(thumbnail)}`
    : `${API}/api/projects/${projectId}/originals/${encodeURIComponent(thumbnail)}`;

  const [src, setSrc] = useState(`${primaryUrl}${cacheBuster}`);
  const [loaded, setLoaded] = useState(false);
  // True when both primary and fallback URLs have errored, used to
  // stop hammering the network and to keep the BlurHash placeholder
  // visible (the only thing the user can see).
  const [bothFailed, setBothFailed] = useState(false);
  useEffect(() => {
    setSrc(`${primaryUrl}${cacheBuster}`);
    setLoaded(false);
    setBothFailed(false);
  }, [primaryUrl, cacheBuster]);

  // Slow-load timeout: if the current src hasn't loaded in 6 s, treat
  // it as failed and walk the fallback chain. Covers the case where
  // the server's lazy-bake is queued behind other renders or the
  // request is just stuck, the image otherwise sits "never rendered"
  // until the user reloads. Cleared whenever onLoad / onError fires
  // so we don't double-advance.
  useEffect(() => {
    if (loaded || bothFailed) return;
    const fb = `${fallbackUrl}${cacheBuster}`;
    const fb2 = `${finalFallbackUrl}${cacheBuster}`;
    const t = window.setTimeout(() => {
      if (src !== fb && src !== fb2) {
        setSrc(fb);
      } else if (src === fb && fb !== fb2) {
        setSrc(fb2);
      } else {
        setBothFailed(true);
      }
    }, 6000);
    return () => window.clearTimeout(t);
  }, [src, loaded, bothFailed, fallbackUrl, finalFallbackUrl, cacheBuster]);

  return (
    <>
      {/* BlurHash placeholder, only renders when the backend
          actually returned a hash. Sized to fill the parent's
          aspect-video slot via the absolute inset wrapper. The
          placeholder is unconditionally beneath the image so the
          fade-in is a clean overlay rather than a swap. Stays
          visible forever when bothFailed (the image is the
          fallback). */}
      {blurhash && (
        <BlurhashCanvas
          hash={blurhash}
          width={32}
          height={18}
          punch={1}
          className="absolute inset-0 h-full w-full"
          // The component renders to an internal canvas at the
          // requested resolution; CSS upscales to fill the slot.
          // 32×18 is plenty since blurhash is a smooth gradient.
          style={{ width: "100%", height: "100%" }}
        />
      )}
      {!bothFailed && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={alt}
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-500 ease-out"
          style={{ opacity: loaded ? 1 : 0 }}
          onLoad={() => setLoaded(true)}
          onError={() => {
            const fb = `${fallbackUrl}${cacheBuster}`;
            const fb2 = `${finalFallbackUrl}${cacheBuster}`;
            if (src !== fb && src !== fb2) {
              // Try the V1/V2-specific fallback URL (annotated
              // preview / V2 reference path).
              setSrc(fb);
              setLoaded(false);
            } else if (src === fb && fb !== fb2) {
              // V1's annotated preview missing, try the original.
              setSrc(fb2);
              setLoaded(false);
            } else {
              // Last fallback also 404'd, give up. Logged so debug
              // can correlate with backend serve errors. Without
              // the bothFailed flag the browser would re-fire
              // onError indefinitely.
              console.warn("[ProjectThumbnail] all URLs failed:", primaryUrl, fallbackUrl, finalFallbackUrl);
              setBothFailed(true);
            }
          }}
        />
      )}
    </>
  );
}


function PublicCard({
  project,
  owner,
  loggedIn,
  onOpen,
  onLike,
  onFavourite,
}: {
  project: ProjectSummary;
  owner: OwnerInfo | null;
  loggedIn: boolean;
  onOpen: () => void;
  onLike: () => void;
  onFavourite: () => void;
}) {
  const hue = hueFor(project.name);
  // Tag-chip colours come from colourForLabelStable now, same label,
  // same colour across every surface (workspace card, public card,
  // project view). hue stays for the thumbnail fallback gradient.
  const handle = project.createdBy || "";
  const displayName = owner?.name || handle || "you";
  const authorInitial = (displayName[0] ?? "?").toUpperCase();
  const statusBadges = projectStatusBadges(project);
  const [tagsOverflowOpen, setTagsOverflowOpen] = useState(false);

  return (
    <article
      onClick={onOpen}
      className="pk-card pk-card-hover group flex h-full flex-col rounded-2xl overflow-hidden cursor-pointer text-left"
      style={{
        position: tagsOverflowOpen ? "relative" : undefined,
        zIndex: tagsOverflowOpen ? 1001 : undefined,
      }}
    >
      <div
        className="aspect-video relative"
        style={{
          backgroundImage: project.thumbnail
            ? undefined
            : `linear-gradient(135deg, hsla(${hue},60%,30%,0.6), hsla(${(hue + 60) % 360},60%,20%,0.6))`,
        }}
      >
        {project.thumbnail ? (
          // Annotated preview when the image has been auto-labelled,
          // raw original otherwise (server returns 404 for the
          // annotated URL on un-labelled images and we swap on
          // error). `?v=<updatedAt>` cache-busts whenever the
          // manifest is rewritten, so manual edits in the editor
          // surface here on the next /api/projects poll.
          <ProjectThumbnail
            projectId={project.id}
            thumbnail={project.thumbnail}
            updatedAt={project.updatedAt}
            v2={!!project.v2}
            blurhash={project.cover_blurhash ?? null}
            alt={project.name}
          />
        ) : (
          <div className="absolute inset-0 grid place-items-center">
            <span className="text-foreground/50 text-sm">No images yet</span>
          </div>
        )}
        {project.certified && (
          <div className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 backdrop-blur-md border border-amber-300/40 text-amber-50 px-2.5 py-1 text-[10px] font-medium shadow-lg">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
              <path d="M12 2l2.39 4.84L20 8l-4 3.9.94 5.5L12 14.77 7.06 17.4 8 11.9 4 8l5.61-1.16L12 2z" />
            </svg>
            Certified
          </div>
        )}
        {/* Frosted white favourite button, matches the workspace card. */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onFavourite();
          }}
          disabled={!loggedIn}
          aria-pressed={project.favouritedByMe}
          title={loggedIn ? (project.favouritedByMe ? "Unfavourite" : "Favourite") : "Login to favourite"}
          className={[
            "absolute top-3 right-3 h-8 w-8 rounded-full grid place-items-center backdrop-blur-md transition-colors",
            !loggedIn
              ? "bg-white/70 text-zinc-400 cursor-not-allowed"
              : project.favouritedByMe
              ? "bg-white/90 hover:bg-white text-amber-500"
              : "bg-white/85 hover:bg-white text-zinc-700",
          ].join(" ")}
          style={{ boxShadow: "0 1px 6px rgb(var(--shadow-rgb) / 0.45), 0 0 0 1px rgb(var(--shadow-rgb) / 0.08)" }}
        >
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4"
            fill={project.favouritedByMe ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
          >
            <path d="M12 17.3l-6.18 3.7 1.64-7.03L2 9.24l7.19-.61L12 2l2.81 6.63L22 9.24l-5.46 4.73L18.18 21z" />
          </svg>
        </button>
      </div>

      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-lg font-semibold tracking-tight truncate flex items-center gap-2">
              <span className="truncate">{project.name}</span>
              {project.private && <PrivateLockIcon />}
              {project.derived && <DerivedBadge parentName={project.derived.parentName} />}
            </h3>
          </div>
          {statusBadges.length > 0 && (
            <div className="flex flex-wrap gap-1.5 justify-end shrink-0 pt-0.5">
              {statusBadges.map((b) => (
                <span
                  key={b.label}
                  className={[
                    "rounded-full px-2 py-0.5 text-[10px] leading-normal uppercase tracking-wider border",
                    b.classes,
                  ].join(" ")}
                >
                  {b.label}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Compact inline stats, matching the workspace card. */}
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
          <div className="flex items-baseline gap-1.5">
            <span className="text-[10px] uppercase tracking-wider text-foreground/40">Images</span>
            <span className="tabular-nums text-sm font-normal text-[var(--foreground)]">{project.n_images.toLocaleString()}</span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-[10px] uppercase tracking-wider text-foreground/40">Labelled</span>
            <span className="tabular-nums text-sm font-normal text-[var(--foreground)]">
              {(project.n_labelled ?? 0).toLocaleString()}
              <span className="text-foreground/35 font-normal"> / {project.n_images.toLocaleString()}</span>
            </span>
          </div>
        </div>

        {/* Always render this row at the same height even when the
            project has no labels yet, keeps the footer row pinned to
            the same vertical position across cards on the public feed. */}
        <ProjectTagsRow
          tags={project.tags ?? []}
          labelAliases={project.label_aliases}
          colourOverrides={project.labelColours}
          onOverflowOpenChange={setTagsOverflowOpen}
        />

        {/* Social footer bar: author byline (left) + like engagement (right),
            pinned to the base of the card with a hairline divider. This is the
            Community card's signature — the workspace dataset cards have no
            author/engagement bar — and mt-auto makes the body fill the card so
            every card lands the same baseline regardless of tag/stat count. */}
        <div className="mt-auto flex items-center justify-between gap-2 border-t border-foreground/[0.06] pt-3 text-xs text-foreground/45">
          <span className="flex min-w-0 items-center gap-1.5">
            {owner?.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={owner.image}
                alt=""
                className="h-5 w-5 rounded-full object-cover shrink-0"
                referrerPolicy="no-referrer"
              />
            ) : (
              <span
                className="h-5 w-5 rounded-full grid place-items-center text-[9px] font-semibold text-[var(--foreground)] shrink-0"
                style={{ backgroundImage: `linear-gradient(135deg, hsl(${hue},70%,55%), hsl(${(hue + 60) % 360},70%,55%))` }}
              >
                {authorInitial}
              </span>
            )}
            <span className="truncate text-foreground/65">{handle ? `@${handle}` : "@you"}</span>
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onLike();
            }}
            disabled={!loggedIn}
            aria-pressed={project.likedByMe}
            className={[
              "inline-flex items-center gap-1.5 rounded-full px-2 py-1 -mr-2 transition-colors shrink-0",
              !loggedIn
                ? "opacity-40 cursor-not-allowed"
                : project.likedByMe
                ? "text-pink-500 hover:text-pink-400"
                : "hover:text-foreground",
            ].join(" ")}
            title={loggedIn ? (project.likedByMe ? "Unlike" : "Like") : "Login to like"}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill={project.likedByMe ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M12 21s-7-4.35-7-10a4 4 0 0 1 7-2.65A4 4 0 0 1 19 11c0 5.65-7 10-7 10z" />
            </svg>
            <span className="tabular-nums">{(project.likes ?? 0).toLocaleString()}</span>
          </button>
        </div>
      </div>
    </article>
  );
}
