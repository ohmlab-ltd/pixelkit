"use client";

// Trim + sample-rate modal for video imports. Ported from V1's
// ProjectView so V2's drop zone (which advertises video/*) actually
// has a destination for dropped videos instead of silently filtering
// them.
//
// Flow: parent queues videos → modal mounts → user picks trim + fps →
// onConfirm fires → parent calls extractVideoFrames (lib/videoFrames)
// and re-feeds the result into the image upload pipeline.

import { useEffect, useRef, useState } from "react";

const THUMB_COUNT = 16;

export function VideoFrameModal({
  file,
  extracting,
  onCancel,
  onConfirm,
}: {
  file: File;
  /** Set while the parent is mid-extraction so the modal swaps to
      its progress view + disables every cancellation affordance. */
  extracting: { done: number; total: number } | null;
  onCancel: () => void;
  onConfirm: (params: { start: number; end: number; fps: number }) => void;
}) {
  // Object URL lifecycle is managed inside the effect (not lazy-init
  // useState) so React 18's StrictMode double-mount doesn't revoke
  // the URL the live <video> element is still pointing at, that bug
  // surfaced as "couldn't decode this video" the instant the modal
  // opened, even on a clean 20 MB MP4.
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [fps, setFps] = useState(1);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [thumbs, setThumbs] = useState<string[]>([]);
  const isExtracting = !!extracting;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isExtracting) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, isExtracting]);

  // Generate the thumbnail strip on a hidden video. Decoupled from
  // the visible <video> so scrubbing/playing the preview doesn't
  // fight the seeks needed for thumbs.
  useEffect(() => {
    if (!loaded || duration <= 0 || !src) return;
    let cancelled = false;
    (async () => {
      const v = document.createElement("video");
      v.src = src;
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      try {
        await new Promise<void>((resolve, reject) => {
          const ok = () => resolve();
          const err = () => reject(new Error("load failed"));
          v.addEventListener("loadeddata", ok, { once: true });
          v.addEventListener("error", err, { once: true });
        });
        const N = THUMB_COUNT;
        const TW = 120;
        const ar = (v.videoHeight || 1) / (v.videoWidth || 1);
        const TH = Math.max(40, Math.round(TW * ar));
        const canvas = document.createElement("canvas");
        canvas.width = TW;
        canvas.height = TH;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const out: string[] = [];
        for (let i = 0; i < N; i++) {
          if (cancelled) return;
          const t = (i / Math.max(1, N - 1)) * duration * 0.999;
          await new Promise<void>((resolve) => {
            let done = false;
            const onSeeked = () => {
              if (done) return;
              done = true;
              v.removeEventListener("seeked", onSeeked);
              resolve();
            };
            v.addEventListener("seeked", onSeeked);
            v.currentTime = t === v.currentTime ? t + 0.001 : t;
            setTimeout(() => {
              if (!done) {
                done = true;
                v.removeEventListener("seeked", onSeeked);
                resolve();
              }
            }, 1500);
          });
          ctx.drawImage(v, 0, 0, TW, TH);
          out.push(canvas.toDataURL("image/jpeg", 0.6));
          if (!cancelled) setThumbs([...out]);
        }
      } catch {
        // Non-fatal, modal still works without the strip.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loaded, duration, src]);

  const onLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const d = e.currentTarget.duration;
    if (Number.isFinite(d) && d > 0) {
      setDuration(d);
      setEnd(d);
      setLoaded(true);
    }
  };

  const seek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return `${m}:${r.toFixed(1).padStart(4, "0")}`;
  };

  const frameCount = duration > 0 ? Math.max(1, Math.floor((end - start) * fps) + 1) : 0;
  const span = end - start;

  return (
    <div
      className="pk-backdrop fixed inset-0 z-[1300] flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      onClick={(e) => {
        if (e.target === e.currentTarget && !isExtracting) onCancel();
      }}
    >
      <div className="pk-glass pk-pop max-w-2xl w-full rounded-2xl overflow-hidden">
        <header className="px-6 pt-6 pb-3 flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-foreground/45">Video import</div>
            <h2 className="mt-1 text-2xl font-medium tracking-tight truncate">
              {file.name}
            </h2>
          </div>
          <button
            onClick={onCancel}
            disabled={isExtracting}
            aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-full text-foreground/60 hover:text-foreground hover:bg-foreground/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span className="text-lg leading-none" aria-hidden="true">×</span>
          </button>
        </header>

        <div className="px-6 pb-2">
          <div className="relative rounded-xl overflow-hidden bg-[var(--background)] border border-foreground/[0.06]">
            {src && (
            /* eslint-disable-next-line jsx-a11y/media-has-caption */
            <video
              ref={videoRef}
              src={src}
              controls
              playsInline
              preload="auto"
              className="w-full max-h-[50vh] block bg-[var(--background)]"
              onLoadedMetadata={onLoadedMetadata}
              onError={(e) => {
                // Only flag a real decode failure. The bare `onError`
                // signature on HTMLVideoElement fires from harmless
                // transient states too (seek-during-buffer, abort on
                // re-render, etc.); only surface the permanent
                // codec-not-supported case.
                const err = (e.currentTarget as HTMLVideoElement).error;
                if (err && (err.code === err.MEDIA_ERR_SRC_NOT_SUPPORTED || err.code === err.MEDIA_ERR_DECODE)) {
                  setLoadError("Couldn't decode this video, try a different format (mp4 / webm / mov).");
                }
              }}
            />
            )}
            {!loaded && !loadError && (
              <div className="absolute inset-0 grid place-items-center bg-black/70 backdrop-blur-sm">
                <div className="flex items-center gap-3 text-sm text-foreground/85">
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="animate-spin">
                    <path d="M21 12a9 9 0 1 1-6.2-8.55" />
                  </svg>
                  Loading video…
                </div>
              </div>
            )}
            {loadError && (
              <div className="absolute inset-0 grid place-items-center bg-black/80 p-6 text-center">
                <p className="text-sm text-rose-200">{loadError}</p>
              </div>
            )}
          </div>
        </div>

        <div className="px-6 py-5 grid gap-5">
          {/* Trim slider, iOS-style two-handle range over a single
              track with thumbnail strip background. */}
          <div>
            <div className="flex items-center justify-between text-xs text-foreground/55 mb-2">
              <span className="flex items-center gap-2">
                Trim
                {loaded && thumbs.length > 0 && thumbs.length < THUMB_COUNT && (
                  <span className="inline-flex items-center gap-1.5 text-foreground/40">
                    <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="animate-spin">
                      <path d="M21 12a9 9 0 1 1-6.2-8.55" />
                    </svg>
                    Generating preview…
                  </span>
                )}
              </span>
              <span className="font-mono tabular-nums">
                {fmt(start)}  →  {fmt(end)}
                <span className="text-foreground/35 ml-2">({fmt(span)})</span>
              </span>
            </div>
            <TrimSlider
              duration={duration}
              start={start}
              end={end}
              thumbs={thumbs}
              disabled={!loaded || isExtracting}
              onChange={(s, e) => {
                setStart(s);
                setEnd(e);
              }}
              onScrub={(t) => seek(t)}
            />
          </div>

          {/* Frame rate */}
          <div>
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-foreground/55">Sample rate</span>
              <span className="font-mono tabular-nums text-sm text-foreground/90">{fps.toFixed(1)} fps</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={10}
              step={0.5}
              value={fps}
              onChange={(e) => setFps(parseFloat(e.target.value))}
              disabled={isExtracting}
              className="w-full accent-white disabled:opacity-50"
            />
            <div className="flex items-center justify-between text-[10px] text-foreground/35 font-mono tabular-nums mt-1">
              <span>0.5</span>
              <span>10</span>
            </div>
          </div>
        </div>

        {isExtracting ? (
          <footer className="px-6 py-4 border-t border-[var(--border)] grid gap-2">
            <div className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-sm text-foreground/85">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-orange-400 opacity-70" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-orange-400" />
                </span>
                Extracting frames
              </span>
              <span className="text-xs text-foreground/55 tabular-nums">
                {extracting!.done} / {extracting!.total} ·{" "}
                {Math.round((extracting!.done / Math.max(1, extracting!.total)) * 100)}%
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-foreground/[0.08]">
              <div
                className="h-full rounded-full transition-all duration-150"
                style={{
                  width: `${Math.min(100, (extracting!.done / Math.max(1, extracting!.total)) * 100)}%`,
                  background: "linear-gradient(90deg, #fb923c, #f97316)",
                  boxShadow: "0 0 14px rgba(249,115,22,0.5)",
                }}
              />
            </div>
            <p className="text-[11px] text-foreground/45">
              Don&rsquo;t close this window, frames are being decoded locally.
            </p>
          </footer>
        ) : (
          <footer className="px-6 py-4 border-t border-[var(--border)] flex items-center justify-between gap-3">
            <span className="text-xs text-foreground/55 font-mono tabular-nums">
              {loaded ? `${frameCount} frame${frameCount === 1 ? "" : "s"}` : "Loading…"}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={onCancel}
                className="rounded-full border border-[var(--border)] px-5 py-2 text-sm hover:border-zinc-500 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onConfirm({ start, end, fps })}
                disabled={!loaded || frameCount === 0 || !!loadError}
                className="rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Extract &amp; upload
              </button>
            </div>
          </footer>
        )}
      </div>
    </div>
  );
}

function TrimSlider({
  duration,
  start,
  end,
  thumbs,
  disabled,
  onChange,
  onScrub,
}: {
  duration: number;
  start: number;
  end: number;
  thumbs: string[];
  disabled?: boolean;
  onChange: (start: number, end: number) => void;
  onScrub?: (t: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef<"start" | "end" | null>(null);

  const positionAt = (clientX: number) => {
    const track = trackRef.current;
    if (!track || duration <= 0) return 0;
    const rect = track.getBoundingClientRect();
    const t = ((clientX - rect.left) / rect.width) * duration;
    return Math.max(0, Math.min(duration, t));
  };

  const onPointerDown = (which: "start" | "end") => (e: React.PointerEvent) => {
    if (disabled) return;
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    draggingRef.current = which;
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    const t = positionAt(e.clientX);
    if (draggingRef.current === "start") {
      const next = Math.min(t, end - 0.05);
      onChange(Math.max(0, next), end);
      onScrub?.(next);
    } else {
      const next = Math.max(t, start + 0.05);
      onChange(start, Math.min(duration, next));
      onScrub?.(next);
    }
  };

  const onPointerUp = () => {
    draggingRef.current = null;
  };

  const startPct = duration > 0 ? (start / duration) * 100 : 0;
  const endPct = duration > 0 ? (end / duration) * 100 : 100;
  const HANDLE_W = 14;

  return (
    <div
      ref={trackRef}
      className={[
        "relative h-20 select-none rounded-xl overflow-hidden border border-foreground/10 bg-[var(--background)]",
        disabled ? "opacity-50" : "",
      ].join(" ")}
      style={{ touchAction: "none" }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <div className="absolute inset-0 flex">
        {(thumbs.length > 0 ? thumbs : Array.from({ length: 16 }).map(() => "")).map((url, i) => (
          <div
            key={i}
            className="flex-1 bg-cover bg-center"
            style={{
              backgroundImage: url ? `url(${url})` : undefined,
              backgroundColor: url ? undefined : "rgb(var(--foreground-rgb) / 0.04)",
              borderLeft: i === 0 ? undefined : "1px solid rgb(var(--shadow-rgb) / 0.25)",
            }}
          />
        ))}
      </div>

      <div
        className="absolute top-0 bottom-0 left-0 bg-black/60 pointer-events-none"
        style={{ width: `${startPct}%` }}
      />
      <div
        className="absolute top-0 bottom-0 right-0 bg-black/60 pointer-events-none"
        style={{ width: `${100 - endPct}%` }}
      />

      <div
        className="absolute top-0 h-1 bg-orange-500 pointer-events-none"
        style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}
      />
      <div
        className="absolute bottom-0 h-1 bg-orange-500 pointer-events-none"
        style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}
      />

      <button
        type="button"
        aria-label="Trim start"
        onPointerDown={onPointerDown("start")}
        className="absolute top-0 bottom-0 cursor-ew-resize touch-none flex items-center justify-center"
        style={{
          left: `${startPct}%`,
          width: HANDLE_W,
          transform: "translateX(-50%)",
          background: "#f97316",
          boxShadow: "0 0 0 1px rgb(var(--shadow-rgb) / 0.25), 0 4px 14px rgba(249,115,22,0.45)",
        }}
      >
        <span className="block h-6 w-[2px] rounded-full bg-foreground/80" />
      </button>

      <button
        type="button"
        aria-label="Trim end"
        onPointerDown={onPointerDown("end")}
        className="absolute top-0 bottom-0 cursor-ew-resize touch-none flex items-center justify-center"
        style={{
          left: `${endPct}%`,
          width: HANDLE_W,
          transform: "translateX(-50%)",
          background: "#f97316",
          boxShadow: "0 0 0 1px rgb(var(--shadow-rgb) / 0.25), 0 4px 14px rgba(249,115,22,0.45)",
        }}
      >
        <span className="block h-6 w-[2px] rounded-full bg-foreground/80" />
      </button>
    </div>
  );
}
