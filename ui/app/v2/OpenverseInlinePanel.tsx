"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import { apiFetch } from "../../lib/apiFetch";
import { GlassDialog } from "./GlassDialog";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

// Ported 1:1 from V1's ProjectView Openverse panel, same UX,
// adapted to V2's manifest shape (imports + /from_urls endpoint).
// Renders as a flat "Import from web" button inside the upload
// panel's action cluster; the dialog holds a search input + 5-image
// preview, prompts for "Yes, this is what I'm looking for", then a
// settings stage with a count slider, then a Pull/review stage.

type ImportImage = {
  url?: string | null;
  thumbnail?: string | null;
  creator?: string | null;
  license?: string | null;
  license_version?: string | null;
  source?: string | null;
  foreign_landing_url?: string | null;
  width?: number | null;
  height?: number | null;
};

const PREVIEW_PAGE_SIZE = 5;
const PREVIEW_FETCH_COUNT = 25;
const REJECTED_KEY = (projectId: string) => `openverse_rejected_v2:${projectId}`;

export function OpenverseInlinePanel({
  projectId,
  alreadyImportedUrls,
  onAdded,
}: {
  projectId: string;
  /** URLs already imported into this project (read from
      manifest.imports[i].source.url). Filtered out of every
      Openverse response so the user doesn't accidentally re-add
      something already in the dataset. */
  alreadyImportedUrls: string[];
  /** Called with the count of newly-added images so the parent
      can refresh its gallery state. */
  onAdded?: (added: number) => Promise<void> | void;
}) {
  const [importOpen, setImportOpen] = useState(false);
  const [importDesc, setImportDesc] = useState("");
  const [importResults, setImportResults] = useState<ImportImage[]>([]);
  const [importLoading, setImportLoading] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importStage, setImportStage] = useState<"search" | "settings" | "review">("search");
  const [importCount, setImportCount] = useState<number>(50);
  const [importBroken, setImportBroken] = useState<Set<string>>(new Set());
  const [importPreviewPage, setImportPreviewPage] = useState(0);
  const [importSearched, setImportSearched] = useState(false);
  const [importPulled, setImportPulled] = useState<ImportImage[]>([]);
  const [importPulling, setImportPulling] = useState(false);
  const [importBadUrls, setImportBadUrls] = useState<Set<string>>(new Set());
  const [importAdding, setImportAdding] = useState(false);

  const [rejectedUrls, setRejectedUrls] = useState<Set<string>>(() => {
    if (typeof window === "undefined") return new Set();
    try {
      const raw = window.localStorage.getItem(REJECTED_KEY(projectId));
      if (!raw) return new Set();
      const arr = JSON.parse(raw);
      return new Set(Array.isArray(arr) ? arr.filter((x) => typeof x === "string") : []);
    } catch {
      return new Set();
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(
        REJECTED_KEY(projectId), JSON.stringify(Array.from(rejectedUrls)),
      );
    } catch {
      /* quota / private mode, keep working in-memory */
    }
  }, [projectId, rejectedUrls]);

  const importedSet = useMemo(
    () => new Set(alreadyImportedUrls.filter((u) => typeof u === "string" && u)),
    [alreadyImportedUrls],
  );
  const excludedUrls = useMemo(() => {
    const s = new Set<string>(rejectedUrls);
    for (const u of importedSet) s.add(u);
    return s;
  }, [rejectedUrls, importedSet]);

  const closeImport = useCallback(() => {
    setImportOpen(false);
    setImportStage("search");
    setImportPulled([]);
    setImportBadUrls(new Set());
    setImportPulling(false);
    setImportSearched(false);
  }, []);

  // Search → fetch + probe thumbnails. Same two-stage probe V1 ships
  // (dimension check + tiny pixel-variance read via canvas) so broken
  // / placeholder / blank thumbnails get filtered before the user
  // ever sees them.
  const searchImportImages = useCallback(async () => {
    const q = importDesc.trim();
    if (!q) return;
    setImportLoading(true);
    setImportError(null);
    setImportResults([]);
    setImportStage("search");
    setImportBroken(new Set());
    setImportPreviewPage(0);
    setImportSearched(false);
    try {
      const params = new URLSearchParams({
        q, count: String(PREVIEW_FETCH_COUNT), commercial: "true",
      });
      const r = await fetch(`${API}/api/openverse/search?${params.toString()}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      const all: ImportImage[] = Array.isArray(d.results) ? d.results : [];
      const seenInBatch = new Set<string>();
      const candidates: ImportImage[] = [];
      for (const c of all) {
        const key = c.url || c.thumbnail || "";
        if (!key || seenInBatch.has(key)) continue;
        if (excludedUrls.has(c.url ?? "")) continue;
        seenInBatch.add(key);
        candidates.push(c);
      }
      const PROBE_TIMEOUT_MS = 5000;
      const CORS_TIMEOUT_MS = 2500;
      const MIN_DIM = 96;
      const MIN_VARIANCE = 8;
      const probe = (img: ImportImage): Promise<ImportImage | null> =>
        new Promise((resolve) => {
          const url = img.thumbnail || img.url;
          if (!url) { resolve(null); return; }
          let settled = false;
          const finish = (ok: boolean, reason?: string) => {
            if (settled) return;
            settled = true;
            window.clearTimeout(stageOneTimer);
            if (!ok && reason) console.debug(`[openverse] dropped ${url}: ${reason}`);
            resolve(ok ? img : null);
          };
          const stage1 = new window.Image();
          const stageOneTimer = window.setTimeout(
            () => finish(false, "load timeout"), PROBE_TIMEOUT_MS,
          );
          stage1.onerror = () => finish(false, "load error");
          stage1.onload = () => {
            if (stage1.naturalWidth < MIN_DIM || stage1.naturalHeight < MIN_DIM) {
              finish(false, `tiny ${stage1.naturalWidth}x${stage1.naturalHeight}`);
              return;
            }
            const stage2 = new window.Image();
            stage2.crossOrigin = "anonymous";
            const corsTimer = window.setTimeout(() => finish(true), CORS_TIMEOUT_MS);
            stage2.onerror = () => { window.clearTimeout(corsTimer); finish(true); };
            stage2.onload = () => {
              window.clearTimeout(corsTimer);
              try {
                const canvas = document.createElement("canvas");
                canvas.width = 32; canvas.height = 32;
                const ctx = canvas.getContext("2d");
                if (!ctx) { finish(true); return; }
                ctx.drawImage(stage2, 0, 0, 32, 32);
                const { data } = ctx.getImageData(0, 0, 32, 32);
                const n = data.length / 4;
                let mR = 0, mG = 0, mB = 0;
                for (let i = 0; i < data.length; i += 4) {
                  mR += data[i]; mG += data[i + 1]; mB += data[i + 2];
                }
                mR /= n; mG /= n; mB /= n;
                let v = 0;
                for (let i = 0; i < data.length; i += 4) {
                  const dr = data[i] - mR;
                  const dg = data[i + 1] - mG;
                  const db = data[i + 2] - mB;
                  v += dr * dr + dg * dg + db * db;
                }
                const stdev = Math.sqrt(v / (n * 3));
                if (stdev < MIN_VARIANCE) {
                  finish(false, `flat thumbnail (stdev ${stdev.toFixed(2)})`);
                  return;
                }
                finish(true);
              } catch {
                finish(true);
              }
            };
            stage2.src = url;
          };
          stage1.src = url;
        });
      const slot: (ImportImage | null | undefined)[] = new Array(candidates.length).fill(undefined);
      let firstPageDone = false;
      const flush = () => {
        const validNow: ImportImage[] = [];
        for (const x of slot) if (x) validNow.push(x);
        setImportResults(validNow);
        if (!firstPageDone && validNow.length >= PREVIEW_PAGE_SIZE) {
          firstPageDone = true;
          setImportLoading(false);
        }
      };
      await Promise.all(candidates.map(async (img, idx) => {
        const result = await probe(img);
        slot[idx] = result;
        flush();
      }));
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportLoading(false);
      setImportSearched(true);
    }
  }, [importDesc, excludedUrls]);

  const pullImportImages = useCallback(async () => {
    const q = importDesc.trim();
    if (!q) return;
    setImportStage("review");
    setImportPulling(true);
    setImportPulled([]);
    setImportBadUrls(new Set());
    try {
      const overCount = Math.min(250, Math.max(importCount, importCount + Math.min(importCount, 25)));
      const params = new URLSearchParams({
        q, count: String(overCount), commercial: "true",
      });
      const r = await fetch(`${API}/api/openverse/search?${params.toString()}`);
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      const all: ImportImage[] = Array.isArray(d.results) ? d.results : [];
      const filtered = all.filter((c) => !c.url || !excludedUrls.has(c.url));
      const seen = new Set<string>();
      const deduped: ImportImage[] = [];
      for (const c of filtered) {
        const key = c.url || c.thumbnail || "";
        if (!key || seen.has(key)) continue;
        seen.add(key);
        deduped.push(c);
      }
      setImportPulled(deduped.slice(0, importCount));
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportPulling(false);
    }
  }, [importDesc, importCount, excludedUrls]);

  const togglePulledBad = useCallback((url: string) => {
    setImportBadUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
    setRejectedUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
  }, []);

  const addPulledToDataset = useCallback(async () => {
    const goodUrls = importPulled
      .map((p) => p.url)
      .filter((u): u is string => !!u && !importBadUrls.has(u));
    if (goodUrls.length === 0) return;
    setImportAdding(true);
    setImportError(null);
    try {
      const r = await apiFetch(`/api/v2/projects/${projectId}/imports/from_urls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: goodUrls, query: importDesc }),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const data = (await r.json()) as { added?: string[] };
      const addedCount = data.added?.length ?? 0;
      await onAdded?.(addedCount);
      closeImport();
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e));
    } finally {
      setImportAdding(false);
    }
  }, [importPulled, importBadUrls, projectId, importDesc, closeImport, onAdded]);

  // Filtered preview window, broken thumbnails dropped, then sliced
  // into pages of 5 with prev/next.
  const valid = importResults.filter((r) => {
    const k = r.url || r.thumbnail || "";
    return !!(r.thumbnail || r.url) && !importBroken.has(k);
  });
  const totalPages = Math.max(1, Math.ceil(valid.length / PREVIEW_PAGE_SIZE));
  const safePage = Math.min(importPreviewPage, totalPages - 1);
  const start = safePage * PREVIEW_PAGE_SIZE;
  const previewSlots: (ImportImage | null)[] = Array.from(
    { length: PREVIEW_PAGE_SIZE },
  ).map((_, i) => valid[start + i] ?? null);
  const hasPrev = safePage > 0;
  const hasNext = safePage < totalPages - 1;

  return (
    <>
      <button
        type="button"
        onClick={() => setImportOpen(true)}
        className="pk-btn"
        title="Search and pull free Creative-Commons images from the web"
      >
        <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M4 14.9A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.24" />
          <path d="M12 12v9" />
          <path d="m8 17 4 4 4-4" />
        </svg>
        Import from web
      </button>

      <GlassDialog open={importOpen} onClose={closeImport} title="Import images from the web" maxWidth="max-w-3xl">
          <div className="flex flex-col">
            <div className="grid gap-2 pt-4">
              <label className="pk-micro" htmlFor="v2-import-desc">
                What images are you looking for?
              </label>
              <div className="flex items-stretch gap-2">
                <input
                  id="v2-import-desc"
                  type="text"
                  value={importDesc}
                  onChange={(e) => setImportDesc(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !importLoading && importDesc.trim()) {
                      e.preventDefault();
                      void searchImportImages();
                    }
                  }}
                  placeholder="e.g. potholes, hard hats, ripe strawberries"
                  className="flex-1 rounded-md border border-[var(--line)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] placeholder:text-foreground/35 outline-none transition-colors focus:border-[var(--line-strong)]"
                />
                <button
                  type="button"
                  onClick={() => void searchImportImages()}
                  disabled={!importDesc.trim() || importLoading}
                  className="pk-btn px-4"
                >
                  {importLoading ? "Searching…" : "Search"}
                </button>
              </div>
              <p className="flex items-center flex-wrap gap-x-3 gap-y-1 text-xs text-foreground/40 mt-1">
                <span className="inline-flex items-center gap-1.5">
                  <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="text-[var(--ok)]">
                    <path d="M5 13l4 4L19 7" />
                  </svg>
                  Sourced from Openverse - Creative Commons licensed, free for commercial use.
                </span>
                {rejectedUrls.size > 0 && (
                  <button
                    type="button"
                    onClick={() => setRejectedUrls(new Set())}
                    className="ml-auto text-[11px] uppercase tracking-wider text-foreground/35 hover:text-foreground transition-colors"
                    title={`${rejectedUrls.size} URL${rejectedUrls.size === 1 ? "" : "s"} remembered as rejected, click to reset for this project`}
                  >
                    Clear remembered ({rejectedUrls.size})
                  </button>
                )}
              </p>
            </div>

            {/* Error chip, animated reveal so the layout doesn't jump. */}
            <div
              className="grid transition-[grid-template-rows] duration-300 ease-out"
              style={{ gridTemplateRows: importError ? "1fr" : "0fr" }}
            >
              <div className="min-h-0 overflow-hidden">
                <div className="mt-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-2 text-xs text-[var(--bad)]">
                  {importError}
                </div>
              </div>
            </div>

            {/* Results section, search preview (5 thumbnails) +
                stage-specific action row underneath. */}
            <div
              className="grid transition-[grid-template-rows] duration-300 ease-out"
              style={{ gridTemplateRows: valid.length > 0 || importLoading || importSearched || importStage === "review" ? "1fr" : "0fr" }}
            >
              <div className="min-h-0 overflow-hidden">
                <div className="mt-4 grid gap-3">
                  {/* Stage: search preview (5 thumbs + arrows) */}
                  {importStage !== "review" && (
                    !importLoading && importSearched && valid.length === 0 ? (
                      <div className="flex flex-col items-center justify-center gap-2 py-8 text-center">
                        <span className="grid h-9 w-9 place-items-center rounded-full border border-foreground/10 text-foreground/40">
                          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <circle cx="11" cy="11" r="7" />
                            <path d="m20 20-3.5-3.5" />
                          </svg>
                        </span>
                        <span className="text-sm text-foreground/65">No images found</span>
                        <span className="text-xs text-foreground/35">Try a different search term.</span>
                      </div>
                    ) : (
                      <div className="relative pb-5">
                        <ul className="grid gap-2 grid-cols-2 sm:grid-cols-5">
                          {importLoading
                            ? Array.from({ length: 5 }).map((_, i) => (
                                <li key={`skel-${i}`} className="aspect-square rounded-md border border-[var(--line)] bg-[var(--panel)] animate-pulse" style={{ animationDelay: `${i * 80}ms` }} />
                              ))
                            : previewSlots.map((img, i) => img == null ? (
                                <li key={`pad-${i}`} className="aspect-square rounded-md border border-[var(--line-soft)] bg-[var(--panel)]" />
                              ) : (
                                <li key={img.url || i} className="group relative aspect-square rounded-md overflow-hidden border border-[var(--line)]">
                                  {/* eslint-disable-next-line @next/next/no-img-element */}
                                  <img
                                    src={img.thumbnail || img.url || ""}
                                    alt=""
                                    className="w-full h-full object-cover"
                                    onError={() => setImportBroken((prev) => {
                                      const next = new Set(prev);
                                      next.add(img.url || img.thumbnail || "");
                                      return next;
                                    })}
                                  />
                                  {(img.creator || img.license) && (
                                    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/80 via-black/30 to-transparent px-2 py-1.5 text-[10px] text-foreground/85 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                                      <span className="truncate">{img.creator || ""}</span>
                                      {img.license && <span className="uppercase tracking-wider text-foreground/65">{img.license}</span>}
                                    </div>
                                  )}
                                </li>
                              ))}
                        </ul>
                        {!importLoading && totalPages > 1 && (
                          <>
                            {hasPrev && (
                              <button type="button" onClick={() => setImportPreviewPage((p) => Math.max(0, p - 1))} aria-label="Previous" className="absolute left-0 top-1/2 -translate-y-1/2 grid h-8 w-8 place-items-center rounded-md border border-[var(--line)] bg-[var(--surface)] text-[var(--foreground)] transition-colors hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                  <path d="m15 18-6-6 6-6" />
                                </svg>
                              </button>
                            )}
                            {hasNext && (
                              <button type="button" onClick={() => setImportPreviewPage((p) => Math.min(totalPages - 1, p + 1))} aria-label="Next" className="absolute right-0 top-1/2 -translate-y-1/2 grid h-8 w-8 place-items-center rounded-md border border-[var(--line)] bg-[var(--surface)] text-[var(--foreground)] transition-colors hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]">
                                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                                  <path d="m9 6 6 6-6 6" />
                                </svg>
                              </button>
                            )}
                            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex items-center gap-1.5">
                              {Array.from({ length: totalPages }).map((_, p) => (
                                <button key={`dot-${p}`} type="button" onClick={() => setImportPreviewPage(p)} aria-label={`Page ${p + 1}`} aria-current={p === safePage} className={["h-1.5 rounded-full transition-all duration-200", p === safePage ? "w-4 bg-foreground/70" : "w-1.5 bg-foreground/20 hover:bg-foreground/40"].join(" ")} />
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    )
                  )}

                  {/* Stage: search → Yes / Try again */}
                  <div
                    className="grid transition-[grid-template-rows] duration-300 ease-out"
                    style={{ gridTemplateRows: !importLoading && valid.length > 0 && importStage === "search" ? "1fr" : "0fr" }}
                  >
                    <div className="min-h-0 overflow-hidden">
                      <div className="flex items-center justify-between gap-3 pt-1">
                        <span className="text-sm text-foreground/60">Happy with these?</span>
                        <div className="flex items-center gap-2">
                          <button type="button" onClick={() => { setImportStage("search"); setImportResults([]); setImportSearched(false); setImportDesc(""); }} className="pk-btn">
                            Try again
                          </button>
                          <button type="button" onClick={() => setImportStage("settings")} className="pk-btn pk-btn-primary">
                            <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <path d="M5 13l4 4L19 7" />
                            </svg>
                            Yes, this is what I&rsquo;m looking for
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Stage: settings → Pull N images */}
                  <div
                    className="grid transition-[grid-template-rows] duration-300 ease-out"
                    style={{ gridTemplateRows: !importLoading && valid.length > 0 && importStage === "settings" ? "1fr" : "0fr" }}
                  >
                    <div className="min-h-0 overflow-hidden">
                      <div className="grid gap-4 pt-3 mt-1 border-t border-foreground/[0.06]">
                        <div className="grid gap-2">
                          <div className="flex items-center justify-between gap-3">
                            <label className="pk-micro" htmlFor="v2-import-count">Number of images to pull</label>
                            <span className="font-mono tabular-nums text-sm text-foreground/80">{importCount}</span>
                          </div>
                          <input id="v2-import-count" type="range" min={10} max={250} step={10} value={importCount} onChange={(e) => setImportCount(parseInt(e.target.value, 10))} className="w-full accent-[var(--accent)]" />
                          <div className="flex items-center justify-between text-[10px] text-foreground/35 font-mono"><span>10</span><span>250</span></div>
                        </div>
                        <div className="flex items-center justify-end gap-2">
                          <button type="button" onClick={() => setImportStage("search")} className="pk-btn">Back</button>
                          <button type="button" onClick={() => void pullImportImages()} className="pk-btn pk-btn-primary tabular-nums">
                            Pull {importCount} images
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Stage: review pulled set */}
                  {importStage === "review" && (
                    <div className="grid gap-3 pt-3 mt-1 border-t border-foreground/[0.06]">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm text-foreground/60">
                          {importPulling
                            ? "Pulling images…"
                            : `${importPulled.length - importBadUrls.size} of ${importPulled.length} selected · click any image to remove it`}
                        </span>
                        <div className="flex items-center gap-2">
                          <button type="button" onClick={() => { setImportStage("settings"); setImportPulled([]); setImportBadUrls(new Set()); }} className="pk-btn">
                            Back
                          </button>
                          <button
                            type="button"
                            onClick={() => void addPulledToDataset()}
                            disabled={importAdding || importPulled.length - importBadUrls.size === 0}
                            className="pk-btn pk-btn-primary tabular-nums"
                          >
                            {importAdding ? "Adding…" : `Add ${importPulled.length - importBadUrls.size} to dataset`}
                          </button>
                        </div>
                      </div>
                      {importPulling ? (
                        <div className="grid gap-2 grid-cols-3 sm:grid-cols-5 md:grid-cols-6">
                          {Array.from({ length: 12 }).map((_, i) => (
                            <div key={`r-skel-${i}`} className="aspect-square rounded-md border border-[var(--line)] bg-[var(--panel)] animate-pulse" style={{ animationDelay: `${i * 60}ms` }} />
                          ))}
                        </div>
                      ) : (
                        <ul className="grid gap-2 grid-cols-3 sm:grid-cols-5 md:grid-cols-6">
                          {importPulled.map((img, i) => {
                            const url = img.url || "";
                            const bad = !!url && importBadUrls.has(url);
                            return (
                              <li key={url || i} className={[
                                "relative aspect-square rounded-md overflow-hidden border-2 transition-all cursor-pointer",
                                bad ? "border-[var(--bad)] opacity-35" : "border-transparent hover:border-[var(--line-strong)]",
                              ].join(" ")} onClick={() => url && togglePulledBad(url)}>
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img src={img.thumbnail || img.url || ""} alt="" className="absolute inset-0 w-full h-full object-cover" loading="lazy" decoding="async" />
                                {bad && (
                                  <div className="absolute inset-0 grid place-items-center">
                                    <span className="rounded-md bg-black/70 font-mono text-[10px] uppercase tracking-wider font-medium text-white px-2 py-0.5">Skip</span>
                                  </div>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
      </GlassDialog>
    </>
  );
}
