"use client";

// Dataset export modal. Animated CSS-keyframe entrance (popIn + fadeIn
// defined in globals.css), light/dark themed via --surface, and the
// content toggles that drive the backend export endpoint:
//   - format: yolo / coco / voc
//   - include_boxes + include_segmentations (at least one required)
//   - exclude_red / exclude_orange (size-class filtering against the
//     project's target input shape)
//   - include_images (originals + augmentations bundled in the zip)
// Shared between V1 ProjectView and the V2 ProjectViewV2Stub so the
// two flows stay aligned with no copy-paste drift.

import { useEffect, useState } from "react";
import { capture } from "./lib/analytics";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  (typeof window !== "undefined" && window.location.port === "3000"
    ? "http://localhost:8001"
    : "");

type ExportFormat = {
  id: "yolo" | "coco" | "voc" | "cvat" | "labelstudio" | "masks";
  name: string;
  blurb: string;
  supportsSegmentation: boolean;
};

const EXPORT_FORMATS: ExportFormat[] = [
  {
    id: "yolo",
    name: "YOLO",
    blurb: "Per-image .txt files with normalised boxes (or YOLOv8-seg polygons). Ultralytics-friendly.",
    supportsSegmentation: true,
  },
  {
    id: "coco",
    name: "COCO",
    blurb: "Microsoft COCO JSON. The standard for most detection + instance-segmentation trainers.",
    supportsSegmentation: true,
  },
  {
    id: "voc",
    name: "Pascal VOC",
    blurb: "Per-image XML annotations. Bounding boxes only, no polygon spec in the format.",
    supportsSegmentation: false,
  },
  {
    id: "cvat",
    name: "CVAT",
    blurb: "CVAT images-1.1 XML - re-import your labels into a CVAT instance for team review.",
    supportsSegmentation: true,
  },
  {
    id: "labelstudio",
    name: "Label Studio",
    blurb: "Task JSON (with a matching labeling config) for importing into Label Studio.",
    supportsSegmentation: true,
  },
  {
    id: "masks",
    name: "PNG masks",
    blurb: "Class-indexed segmentation masks, one PNG per image, plus labels.txt. Needs polygons.",
    supportsSegmentation: true,
  },
];

export function ExportModal({
  projectId,
  projectName,
  inputShape = "256x256",
  onClose,
}: {
  projectId: string;
  projectName: string;
  inputShape?: string;
  onClose: () => void;
}) {
  const [picked, setPicked] = useState<ExportFormat["id"]>("yolo");
  const [includeImages, setIncludeImages] = useState<boolean>(true);
  const [includeBoxes, setIncludeBoxes] = useState<boolean>(true);
  const [includeSegmentations, setIncludeSegmentations] = useState<boolean>(true);
  const [excludeRed, setExcludeRed] = useState<boolean>(true);
  const [excludeOrange, setExcludeOrange] = useState<boolean>(false);
  // Train fraction (0–100). 80/20 is the conventional default; the backend
  // splits each image deterministically by sha1(filename) so re-exports
  // always place the same image in the same set.
  const [trainPct, setTrainPct] = useState<number>(80);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Export progress. Big projects spend ~all the time on the server
  // packaging the zip, so until bytes start flowing back we show an
  // elapsed-seconds counter + indeterminate bar; once response.body
  // is streaming, we flip to a bytes-downloaded readout.
  const [phase, setPhase] = useState<"building" | "downloading" | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [bytesReceived, setBytesReceived] = useState(0);
  const [bytesTotal, setBytesTotal] = useState<number | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // Don't let Esc dismiss the modal mid-build. The export is still
      // running on the server even if the modal goes away, but losing
      // the only place that shows progress is confusing.
      if (busy) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, busy]);

  // Tick the elapsed-seconds counter every 500 ms while busy so the
  // user can see something is actually happening on long builds.
  useEffect(() => {
    if (!busy) return;
    const startedAt = Date.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startedAt) / 1000));
    }, 500);
    return () => window.clearInterval(id);
  }, [busy]);

  // beforeunload guard. Browsers ignore the custom string but still
  // surface a generic "Leave site?" prompt, which is exactly what
  // we want here, the export is running on the server and closing
  // the tab aborts the download.
  useEffect(() => {
    if (!busy) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "Your export is still building. Closing this tab will cancel it.";
      return e.returnValue;
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [busy]);

  const fmt = EXPORT_FORMATS.find((f) => f.id === picked) ?? EXPORT_FORMATS[0];
  const fmtSupportsSeg = fmt.supportsSegmentation;
  // VOC can't carry polygons, so the seg toggle is effectively a no-op
  // when VOC is picked. Show it disabled (rather than hidden) so the
  // user understands the constraint instead of silently being ignored.
  // PNG masks are the inverse: they're BUILT from segmentations, so the
  // toggle is forced on (the engine 400s a masks export without them).
  const effectiveSeg = picked === "masks" || (includeSegmentations && fmtSupportsSeg);
  const canExport = includeBoxes || effectiveSeg;

  const triggerExport = async () => {
    if (busy || !canExport) return;
    capture("dataset_export", { format: picked, project_id: projectId });
    setBusy(true);
    setError(null);
    setPhase("building");
    setBytesReceived(0);
    setBytesTotal(null);
    try {
      const params = new URLSearchParams({
        format: picked,
        include_images: String(includeImages),
        include_boxes: String(includeBoxes),
        include_segmentations: String(effectiveSeg),
        exclude_red: String(excludeRed),
        exclude_orange: String(excludeOrange),
        input_shape: inputShape || "256x256",
        train_split: String(Math.max(0, Math.min(1, trainPct / 100))),
      });
      const url = `${API}/api/projects/${projectId}/export?${params.toString()}`;
      const r = await fetch(url);
      if (!r.ok) {
        let detail = `http ${r.status}`;
        try {
          const body = await r.json();
          detail = (body && (body.detail || body.error)) || detail;
        } catch {
          // not JSON, leave detail as-is
        }
        throw new Error(detail);
      }
      // Response headers landed → the server has finished building and
      // is now streaming the zip. Flip the phase + start counting
      // bytes so the readout switches from "Building…" to a download
      // progress that ticks.
      setPhase("downloading");
      const lenHeader = r.headers.get("content-length");
      const total = lenHeader ? parseInt(lenHeader, 10) : NaN;
      if (Number.isFinite(total) && total > 0) setBytesTotal(total);
      const cd = r.headers.get("content-disposition") || "";
      const match = /filename="([^"]+)"/.exec(cd);
      const filename = match ? match[1] : `export-${picked}.zip`;

      // Stream the body so we can show bytes-received progress on
      // large exports. Falls back to r.blob() when streaming isn't
      // available (older browsers, polyfilled fetch).
      let blob: Blob;
      if (r.body && typeof r.body.getReader === "function") {
        const reader = r.body.getReader();
        const chunks: BlobPart[] = [];
        let received = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.byteLength;
            setBytesReceived(received);
          }
        }
        blob = new Blob(chunks, { type: r.headers.get("content-type") || "application/zip" });
      } else {
        blob = await r.blob();
        setBytesReceived(blob.size);
      }

      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objUrl);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setPhase(null);
    }
  };

  const formatSeconds = (s: number): string => {
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const r = s % 60;
    return r ? `${m}m ${r}s` : `${m}m`;
  };
  const formatBytes = (n: number): string => {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };
  const downloadPct =
    phase === "downloading" && bytesTotal && bytesTotal > 0
      ? Math.min(100, Math.round((bytesReceived / bytesTotal) * 100))
      : null;

  return (
    <div
      className="pk-backdrop fixed inset-0 z-[1300] flex items-center justify-center overflow-auto p-6"
      role="dialog"
      aria-modal="true"
      onClick={() => { if (!busy) onClose(); }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="pk-glass pk-pop w-full max-w-2xl mt-8 mb-8 rounded-md overflow-hidden"
      >
        <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--line-soft)]">
          <div>
            <div className="pk-micro">Export</div>
            <h2 className="mt-0.5 text-[15px] font-medium tracking-tight">{projectName}</h2>
          </div>
          <button
            onClick={() => { if (!busy) onClose(); }}
            className="text-2xl leading-none px-2 text-[var(--muted)] hover:text-foreground disabled:opacity-40"
            disabled={busy}
            aria-label="close"
          >
            ×
          </button>
        </header>

        <section className="px-6 py-5 grid gap-3">
          <SectionLabel>Format</SectionLabel>
          <div className="grid sm:grid-cols-3 gap-2">
            {EXPORT_FORMATS.map((f) => {
              const active = f.id === picked;
              return (
                <button
                  key={f.id}
                  onClick={() => setPicked(f.id)}
                  className={[
                    "text-left rounded-md border px-4 py-3 transition-colors",
                    active
                      ? "border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--foreground)]"
                      : "border-[var(--line)] bg-[var(--panel)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-medium text-[var(--foreground)]">{f.name}</span>
                  </div>
                  <p className="mt-1 text-[11px] text-foreground/55 leading-snug">{f.blurb}</p>
                </button>
              );
            })}
          </div>
        </section>

        <section className="px-6 pb-2 grid gap-2">
          <SectionLabel>Annotations to include</SectionLabel>
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={includeBoxes}
              onChange={(e) => setIncludeBoxes(e.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            <span className="text-sm text-foreground/85">Bounding boxes</span>
          </label>
          <label
            className={[
              "flex items-center gap-3 select-none",
              fmtSupportsSeg && picked !== "masks"
                ? "cursor-pointer"
                : "cursor-not-allowed opacity-50",
            ].join(" ")}
          >
            <input
              type="checkbox"
              checked={effectiveSeg}
              disabled={!fmtSupportsSeg || picked === "masks"}
              onChange={(e) => setIncludeSegmentations(e.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            <span className="text-sm text-foreground/85">
              Segmentations
              {!fmtSupportsSeg && (
                <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--fg-dim)]">
                  not supported in {fmt.name}
                </span>
              )}
              {picked === "masks" && (
                <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--fg-dim)]">
                  required for PNG masks
                </span>
              )}
            </span>
          </label>
          {!canExport && (
            <p className="text-[11px] text-[var(--warn)] mt-1">
              At least one of boxes or segmentations must be enabled to export.
            </p>
          )}
        </section>

        <section className="px-6 py-4 grid gap-2">
          <SectionLabel>Filter by detection size</SectionLabel>
          <p className="text-[11px] text-foreground/45 -mt-1">
            Boxes are classified against your target input shape ({inputShape || "256x256"}).
            Red boxes are too small for the model to detect; orange boxes are borderline.
          </p>
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={excludeRed}
              onChange={(e) => setExcludeRed(e.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            <span className="text-sm text-foreground/85">
              <span className="inline-block h-2 w-2 rounded-full bg-[hsl(0,78%,56%)] mr-2 align-middle" />
              Exclude red boxes
            </span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={excludeOrange}
              onChange={(e) => setExcludeOrange(e.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            <span className="text-sm text-foreground/85">
              <span className="inline-block h-2 w-2 rounded-full bg-[hsl(38,92%,58%)] mr-2 align-middle" />
              Exclude orange boxes
            </span>
          </label>
        </section>

        <section className="px-6 py-4 grid gap-2">
          <SectionLabel>Train / val split</SectionLabel>
          <div className="flex items-center gap-4">
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={trainPct}
              onChange={(e) => setTrainPct(Number(e.target.value))}
              className="flex-1 accent-[var(--accent)]"
            />
            <span className="text-sm font-mono tabular-nums text-foreground/85 w-28 text-right">
              {trainPct}% / {100 - trainPct}%
            </span>
          </div>
          <p className="text-[11px] text-foreground/45">
            Each image is hashed by filename so the same picture always lands in the same set,
            even across re-exports. Augmentations follow their source image.
          </p>
        </section>

        <section className="px-6 pb-5 grid gap-2">
          <SectionLabel>Bundle</SectionLabel>
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={includeImages}
              onChange={(e) => setIncludeImages(e.target.checked)}
              className="h-4 w-4 accent-[var(--accent)]"
            />
            <span className="text-sm text-foreground/85">Include image files (originals + augmentations)</span>
          </label>
          <p className="text-[11px] text-foreground/40 -mt-1">
            Off: annotations only (filenames reference your local images). On: every photo bundled in the same archive,
            augmentations named <span className="font-mono">aug_&lt;name&gt;_&lt;n&gt;.jpg</span>.
          </p>
        </section>

        {error && (
          <div className="mx-6 mb-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-xs text-[var(--bad)]">
            {error}
          </div>
        )}

        {/* In-flight progress card. Big projects sit in "Building" for
            a long time; surface elapsed seconds + an indeterminate bar
            so the user sees motion. Once response bytes start flowing
            we flip to a deterministic download bar with byte counts.
            A standing warning across both phases tells the user not
            to close the window, the build is happening on the server
            and the request is what carries the zip back. */}
        {busy && (
          <div className="mx-6 mb-4 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-3.5">
            <div className="flex items-center justify-between gap-3 mb-2">
              <div className="flex items-center gap-2.5 min-w-0">
                <span className="inline-block h-3 w-3 rounded-full border-2 border-foreground/20 border-t-[var(--accent)] animate-spin shrink-0" />
                <span className="text-[13px] font-medium text-[var(--foreground)] tabular-nums">
                  {phase === "downloading"
                    ? downloadPct !== null
                      ? `Downloading… ${downloadPct}%`
                      : "Downloading…"
                    : "Building your export…"}
                </span>
              </div>
              <span className="text-[11px] font-mono tabular-nums text-foreground/55 shrink-0">
                {phase === "downloading" && bytesReceived > 0
                  ? bytesTotal
                    ? `${formatBytes(bytesReceived)} / ${formatBytes(bytesTotal)}`
                    : formatBytes(bytesReceived)
                  : formatSeconds(elapsedSec)}
              </span>
            </div>
            <div className="h-1.5 w-full bg-foreground/[0.06] rounded-full overflow-hidden relative">
              {phase === "downloading" && downloadPct !== null ? (
                <div
                  className="h-full bg-[var(--accent)] rounded-full transition-[width] duration-300 ease-out"
                  style={{ width: `${downloadPct}%` }}
                />
              ) : (
                <div className="indeterminate-bar absolute inset-y-0 w-1/3 bg-[var(--accent)] rounded-full" />
              )}
            </div>
            <p className="mt-2.5 text-[11px] text-foreground/55 leading-relaxed">
              <span className="font-medium text-foreground/80">Don&rsquo;t close this tab.</span>
              {" "}Large projects can take a few minutes to package on the server. Leaving the page cancels the build and the download won&rsquo;t arrive.
            </p>
          </div>
        )}

        <footer className="flex items-center justify-between gap-3 px-6 py-4 border-t border-[var(--line-soft)] bg-[var(--panel)]">
          <span className="text-[11px] text-foreground/45">
            Exporting as <span className="font-medium text-foreground/85">{fmt.name}</span>
            {includeImages && <span className="text-foreground/45"> · with images</span>}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              disabled={busy}
              className="rounded-md border border-[var(--line)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] px-4 py-2 text-[13px] text-[var(--fg-soft)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title={busy ? "Cancel disabled while the export is building" : undefined}
            >
              Cancel
            </button>
            <button
              onClick={triggerExport}
              disabled={busy || !canExport}
              className="rounded-md bg-[var(--accent)] text-[var(--accent-contrast)] px-5 py-2 text-[13px] font-medium hover:brightness-105 disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {busy && (
                <span className="inline-block h-3 w-3 rounded-full border-2 border-current border-t-transparent opacity-70 animate-spin" />
              )}
              {busy
                ? phase === "downloading"
                  ? "Downloading…"
                  : `Building… ${formatSeconds(elapsedSec)}`
                : "Export"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="pk-micro mb-1">
      {children}
    </div>
  );
}
