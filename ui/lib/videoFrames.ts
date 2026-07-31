// Client-side video → JPEG-frame extraction. A hidden <video> seeks
// to each target time, draws to a canvas, encodes JPEG, and returns
// File objects with the project's "{name}_frame####.jpg" naming so
// they slot into the existing image-upload flow without any backend
// changes.
//
// Lifted verbatim from V1's ProjectView so both the V1 and V2 drop
// flows can share the same kernel.

export type VideoExtractParams = {
  start: number;
  end: number;
  fps: number;
};

// Per-frame size + quality ladder, matches lib/resize.ts's defaults so
// extracted frames never balloon past what the upload pipeline allows
// downstream. Encoding under the cap inline (rather than at full
// quality then re-compressing in handleImportFiles) keeps a long
// extraction's peak memory bounded — a 1000-frame, 4K video at q=0.92
// is ~2 GB of intermediate Blobs.
const FRAME_MAX_BYTES = 50 * 1024;
const FRAME_MAX_EDGE = 1500;
const FRAME_QUALITY_LADDER = [0.85, 0.72, 0.6, 0.48, 0.36, 0.25];

async function encodeFrameUnderCap(
  canvas: HTMLCanvasElement,
  fname: string,
): Promise<File | null> {
  for (const q of FRAME_QUALITY_LADDER) {
    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", q),
    );
    if (!blob) continue;
    if (blob.size <= FRAME_MAX_BYTES) {
      return new File([blob], fname, { type: "image/jpeg" });
    }
  }
  // Quality floor still over budget → halve dimensions and fall back
  // to a mid-quality encode. Captures the dense-detail outlier
  // without making every frame pay the second pass.
  const half = document.createElement("canvas");
  half.width = Math.max(1, Math.round(canvas.width / 2));
  half.height = Math.max(1, Math.round(canvas.height / 2));
  const ctx = half.getContext("2d");
  if (ctx) {
    ctx.drawImage(canvas, 0, 0, half.width, half.height);
    const fallback: Blob | null = await new Promise((resolve) =>
      half.toBlob(resolve, "image/jpeg", 0.5),
    );
    if (fallback) return new File([fallback], fname, { type: "image/jpeg" });
  }
  return null;
}

export async function extractVideoFrames(
  file: File,
  params: VideoExtractParams,
  onProgress: (i: number, total: number) => void,
): Promise<File[]> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  try {
    await new Promise<void>((resolve, reject) => {
      const onLoaded = () => {
        video.removeEventListener("loadeddata", onLoaded);
        resolve();
      };
      const onErr = () => {
        video.removeEventListener("error", onErr);
        reject(new Error("Failed to load video"));
      };
      video.addEventListener("loadeddata", onLoaded);
      video.addEventListener("error", onErr);
    });

    // Downscale to FRAME_MAX_EDGE on the long side BEFORE encoding —
    // a 4K source frame at full res would never hit the 50 KB cap
    // without falling through to the quality-floor pass, so cap once
    // up front and the quality ladder has a fighting chance on
    // pass 1.
    const srcW = video.videoWidth || 1920;
    const srcH = video.videoHeight || 1080;
    const longest = Math.max(srcW, srcH);
    const scale = longest > FRAME_MAX_EDGE ? FRAME_MAX_EDGE / longest : 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(srcW * scale));
    canvas.height = Math.max(1, Math.round(srcH * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");

    const baseName = file.name.replace(/\.[^.]+$/, "").replace(/\s+/g, "_");
    const interval = 1 / Math.max(0.1, params.fps);
    const span = Math.max(0, params.end - params.start);
    const totalFrames = Math.max(1, Math.floor(span * params.fps) + 1);
    const out: File[] = [];

    for (let i = 0; i < totalFrames; i++) {
      const t = Math.min(params.end, params.start + i * interval);
      await new Promise<void>((resolve) => {
        let done = false;
        const onSeeked = () => {
          if (done) return;
          done = true;
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        // Some browsers ignore tiny seeks if the time matches the
        // current frame exactly, so always nudge by a hair.
        video.currentTime = t === video.currentTime ? t + 0.001 : t;
        // Belt-and-braces timeout, if the browser fails to fire
        // `seeked` (rare, but happens on some codecs), don't hang.
        setTimeout(() => {
          if (!done) {
            done = true;
            video.removeEventListener("seeked", onSeeked);
            resolve();
          }
        }, 3000);
      });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const fname = `${baseName}_frame${String(i + 1).padStart(4, "0")}.jpg`;
      const framed = await encodeFrameUnderCap(canvas, fname);
      if (framed) out.push(framed);
      onProgress(i + 1, totalFrames);
    }

    return out;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Hard cap on individual video files. Browser-side decode + seek of
// anything bigger gets sluggish or OOMs on smaller machines. 100 MB
// has tested OK on a mid-range laptop; bump cautiously past that.
export const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
