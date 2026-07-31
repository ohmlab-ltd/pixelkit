"use client";

import { useEffect, useRef, useState } from "react";
import { containsProfanity } from "./profanity";
import { apiFetch } from "@/lib/apiFetch";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type ResultLite = { image: string; pending?: boolean };

// Allow-list of users who get the "Private project" toggle. Hidden from
// everyone else so the option doesn't clutter the settings panel, the
// backend doesn't care who toggles it, this is a UX gate only.
const PRIVATE_TOGGLE_USERS = ["hamish", "mukund", "faizan"];

type Props = {
  name: string;            // project UUID, used for the API path
  displayName: string;     // human-readable name shown in the input
  cover: string | null;
  results: ResultLite[];
  username: string;        // current viewer; gates the private toggle
  initialPrivate?: boolean;
  onRenamed: (newName: string) => void;
  onCoverChange: (cover: string | null) => void;
  onPrivateChange?: (next: boolean) => void;
  onClose: () => void;
};

export function ProjectSettings({
  name,
  displayName,
  cover,
  results,
  username,
  initialPrivate = false,
  onRenamed,
  onCoverChange,
  onPrivateChange,
  onClose,
}: Props) {
  const [newName, setNewName] = useState(displayName);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPrivate, setIsPrivate] = useState<boolean>(initialPrivate);
  const [savingPrivacy, setSavingPrivacy] = useState(false);
  // Privacy-toggle errors render INSIDE the Visibility section so the
  // user sees them next to the toggle they just touched. The shared
  // `error` state is used by the Rename section and was too far away
  // to associate with a failed toggle, the user reported "the toggle
  // doesn't toggle" because they never saw the rollback message.
  const [privacyError, setPrivacyError] = useState<string | null>(null);
  const canTogglePrivate = PRIVATE_TOGGLE_USERS.includes(username.toLowerCase());

  // Progressive cover-picker rendering, start small so the dialog
  // opens instantly on big projects, then load another batch as the
  // user scrolls the picker grid down. Same IntersectionObserver
  // pattern used on the main image grid.
  const COVER_PAGE_SIZE = 20;
  const [coverLimit, setCoverLimit] = useState(COVER_PAGE_SIZE);
  const coverSentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = coverSentinelRef.current;
    if (!node) return;
    if (coverLimit >= results.length) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setCoverLimit((cur) => Math.min(cur + COVER_PAGE_SIZE, results.length));
          }
        }
      },
      // Use the inner scroll container as the root so the sentinel
      // triggers on grid-internal scroll, not the document scroll.
      { root: node.parentElement?.parentElement ?? null, rootMargin: "200px" },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, [coverLimit, results.length]);

  const togglePrivate = async (next: boolean) => {
    if (savingPrivacy) return;
    const prev = isPrivate;
    setIsPrivate(next);
    setSavingPrivacy(true);
    setPrivacyError(null);
    try {
      const r = await apiFetch(`/api/projects/${name}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ private: next }),
      });
      if (!r.ok) {
        const body = await r.text().catch(() => "");
        // Map the common owner-gate failure to copy a user can actually
        // act on. Anything else surfaces verbatim so we can diagnose
        // from a screenshot.
        let message = body || `http ${r.status}`;
        if (r.status === 403) {
          message =
            "Only the project owner can change visibility. If you're signed in with the right account and still see this, the project may pre-date the ownership-required era; contact support to claim it.";
        }
        throw new Error(message);
      }
      onPrivateChange?.(next);
    } catch (e) {
      // Console-log the raw error so support can read it off a
      // screenshot, then revert the optimistic update and surface
      // the friendly message inline next to the toggle.
      console.warn("[settings/private] toggle failed:", e);
      setIsPrivate(prev);
      setPrivacyError(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingPrivacy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const rename = async () => {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === displayName) {
      onClose();
      return;
    }
    // Block client-side first so the user gets instant feedback and
    // we don't even round-trip to the server. The backend has the
    // same gate as a defence-in-depth.
    const bad = containsProfanity(trimmed);
    if (bad) {
      setError(`"${trimmed}" can't be used as a project name.`);
      // Snap the input back to the current name so the rejected
      // value isn't sitting there waiting to be re-submitted.
      setNewName(displayName);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await apiFetch(`/api/projects/${name}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!r.ok) {
        // Try to extract a clean message from the standard FastAPI
        // `{"detail": "..."}` shape. Fall back to raw text only if
        // the body isn't JSON.
        let msg = `http ${r.status}`;
        const body = await r.text();
        try {
          const parsed = JSON.parse(body);
          if (parsed && typeof parsed.detail === "string") msg = parsed.detail;
          else if (body) msg = body;
        } catch {
          if (body) msg = body;
        }
        // Server-side profanity rejection ends up here when something
        // slipped past the client-side check (different list, etc.).
        // Either way, snap the input back to the current name.
        setNewName(displayName);
        // Friendly rewording of the backend's machine-form message.
        if (/banned term/i.test(msg)) {
          setError(`"${trimmed}" can't be used as a project name.`);
        } else {
          setError(msg);
        }
        return;
      }
      const data = await r.json();
      onRenamed(data.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-40 backdrop-blur-md bg-black/85 flex items-start justify-center overflow-auto p-6" role="dialog" aria-modal="true">
      <div className="bg-[var(--background)] rounded-2xl border border-[var(--border)] max-w-3xl w-full mt-8 mb-8">
        <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
          <h2 className="text-lg font-semibold">Project settings</h2>
          <button
            onClick={onClose}
            className="text-2xl leading-none px-2 text-[var(--muted)] hover:text-foreground"
            aria-label="close"
          >
            ×
          </button>
        </header>

        <section className="px-6 py-5 border-b border-[var(--border)] grid gap-3">
          <label className="text-xs text-[var(--muted)] uppercase tracking-wider">Name</label>
          <div className="flex gap-3 items-center">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") rename();
              }}
              className="flex-1 rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-base focus:outline-none focus:border-[var(--foreground)]/40"
            />
            <button
              onClick={rename}
              disabled={busy || !newName.trim() || newName.trim() === displayName}
              className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {busy ? "Renaming…" : "Rename"}
            </button>
          </div>
          <p className="text-xs text-[var(--muted)]">Display name only, any characters allowed.</p>
          {error && <div className="text-xs text-red-400">{error}</div>}
        </section>

        {canTogglePrivate && (
          <section className="px-6 py-5 border-b border-[var(--border)] grid gap-3">
            <label className="text-xs text-[var(--muted)] uppercase tracking-wider">Visibility</label>
            <button
              type="button"
              role="switch"
              aria-checked={isPrivate}
              disabled={savingPrivacy}
              onClick={() => togglePrivate(!isPrivate)}
              className={[
                "inline-flex items-center gap-3 self-start rounded-full border px-3 py-1.5 text-sm transition-colors",
                isPrivate
                  ? "border-amber-300/40 bg-amber-300/[0.08] text-amber-100 hover:bg-amber-300/[0.12]"
                  : "border-foreground/15 bg-foreground/5 text-foreground/80 hover:bg-foreground/10 hover:text-foreground",
                savingPrivacy ? "opacity-60 cursor-wait" : "",
              ].join(" ")}
            >
              <span
                aria-hidden
                className={[
                  "h-4 w-7 rounded-full p-0.5 transition-colors flex",
                  isPrivate ? "bg-amber-300/70 justify-end" : "bg-foreground/15",
                ].join(" ")}
              >
                <span className="h-3 w-3 rounded-full bg-[#141416]" />
              </span>
              {isPrivate ? "Private, only you can see this project" : "Public, visible in the community feed"}
            </button>
            {privacyError && (
              <div className="text-[12px] text-red-500 dark:text-red-300 max-w-md leading-relaxed">
                {privacyError}
              </div>
            )}
          </section>
        )}

        <section className="px-6 py-5">
          <div className="flex items-center justify-between mb-3">
            <label className="text-xs text-[var(--muted)] uppercase tracking-wider">Cover image</label>
            {cover && (
              <button
                onClick={() => onCoverChange(null)}
                className="text-[10px] uppercase tracking-wider text-[var(--muted)] hover:text-foreground"
              >
                Reset to first image
              </button>
            )}
          </div>
          {results.length === 0 ? (
            <p className="text-sm text-[var(--muted)]">Add some images first.</p>
          ) : (
            <div className="max-h-[55vh] overflow-auto pr-1">
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                {/* Render the first chunk and stream more in as the
                    user scrolls, instant first paint on projects
                    with hundreds of images. */}
                {results.slice(0, coverLimit).map((r) => {
                  const active = cover === r.image;
                  return (
                    <button
                      key={r.image}
                      onClick={() => onCoverChange(r.image)}
                      className={[
                        "relative aspect-square rounded-lg overflow-hidden border-2 transition-colors",
                        active ? "border-[var(--foreground)]" : "border-transparent hover:border-zinc-500",
                      ].join(" ")}
                      title={r.image}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`${API}/api/projects/${name}/originals/${encodeURIComponent(r.image)}`}
                        alt=""
                        className="w-full h-full object-cover"
                        loading="lazy"
                        decoding="async"
                      />
                      {active && (
                        <div className="absolute bottom-1 right-1 rounded-full bg-foreground text-background text-[10px] font-mono px-1.5 py-0.5">
                          cover
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
              {coverLimit < results.length && (
                <>
                  <div ref={coverSentinelRef} aria-hidden="true" className="h-1" />
                  <p className="mt-2 text-[11px] text-[var(--muted)] tabular-nums text-center">
                    Loading more… ({coverLimit} of {results.length})
                  </p>
                </>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
