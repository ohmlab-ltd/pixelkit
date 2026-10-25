// JPEG frames from a video or an .mjpeg recording. A recording is JPEG
// frames end to end, with no container or timing.

// For a video, start and end are seconds, sampled at fps. For a recording,
// they are frame indices, keeping every step-th frame.
export type VideoExtractParams =
  | { start: number; end: number; fps: number }
  | { start: number; end: number; step: number };

export function isMjpeg(file: File): boolean {
  return /\.mjpe?g$/i.test(file.name);
}

// What the video import takes: videos, and recordings from the board.
export function isVideoFile(file: File): boolean {
  return file.type.startsWith("video/") || isMjpeg(file);
}

// Video types the file picker offers. Frames are read through Chromium's
// <video>, which cannot decode AVI, so AVI is left out.
export const VIDEO_ACCEPT = ".mp4,.m4v,.mov,.webm,.mkv,.mjpeg,.mjpg";

export function extractCount(p: VideoExtractParams): number {
  if ("step" in p) return Math.floor((Math.round(p.end) - Math.round(p.start)) / Math.max(1, Math.round(p.step))) + 1;
  return Math.max(1, Math.floor((p.end - p.start) * p.fps) + 1);
}

// The next start of image at or after `from`, or b.length.
function nextStart(b: Uint8Array, from: number): number {
  for (let i = from; i + 2 < b.length; i++) {
    if (b[i] === 0xff && b[i + 1] === 0xd8 && b[i + 2] === 0xff) return i;
  }
  return b.length;
}

// jpegEnd without an end: CUT when the buffer ends inside the frame, BAD
// when the bytes are not JPEG marker structure.
const CUT = -2;
const BAD = -1;

// Exclusive end of the JPEG at start, or CUT or BAD.
function jpegEnd(b: Uint8Array, start: number): number {
  let p = start + 2;
  while (p + 1 < b.length) {
    if (b[p] !== 0xff) return BAD;
    const m = b[p + 1];
    if (m === 0xff) {
      p += 1; // fill byte before a marker
      continue;
    }
    if (m === 0xd9) return p + 2; // end of image
    if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
      p += 2; // markers without a length
      continue;
    }
    if (p + 3 >= b.length) return CUT;
    p += 2 + ((b[p + 2] << 8) | b[p + 3]);
    if (m !== 0xda) continue;
    // Start of scan: entropy-coded data runs to the next marker, which
    // is an 0xFF not followed by a stuffed 0x00 or a restart marker.
    while (p + 1 < b.length && !(b[p] === 0xff && b[p + 1] !== 0 && (b[p + 1] < 0xd0 || b[p + 1] > 0xd7))) p++;
  }
  return CUT;
}

// Read size, and the size past which a cut frame counts as damage.
const SCAN_CHUNK_BYTES = 8 * 1024 * 1024;
const MAX_FRAME_BYTES = 8 * 1024 * 1024;

// [start, end) ranges of JPEGs stored end to end, read in chunks. Walks marker
// segments, as an EXIF thumbnail has its own end marker. Damage is skipped. A
// frame cut by the end of a read is finished with the next.
export async function mjpegFrameRanges(
  file: Blob,
  chunkBytes = SCAN_CHUNK_BYTES,
): Promise<Array<[number, number]>> {
  const out: Array<[number, number]> = [];
  let buf = new Uint8Array(0);
  let base = 0; // file offset of buf[0]
  let read = 0;
  while (read < file.size) {
    const chunk = new Uint8Array(await file.slice(read, read + chunkBytes).arrayBuffer());
    if (chunk.length === 0) break;
    read += chunk.length;
    const joined = new Uint8Array(buf.length + chunk.length);
    joined.set(buf);
    joined.set(chunk, buf.length);
    buf = joined;
    const final = read >= file.size;

    // A start of image may straddle this read and the next.
    let keepFrom = Math.max(0, buf.length - 2);
    let i = nextStart(buf, 0);
    while (i < buf.length) {
      const end = jpegEnd(buf, i);
      if (end >= 0) {
        out.push([base + i, base + end]);
        i = nextStart(buf, end);
        continue;
      }
      if (end === CUT && !final && buf.length - i <= MAX_FRAME_BYTES) {
        keepFrom = i;
        break;
      }
      i = nextStart(buf, i + 2);
    }
    buf = buf.slice(keepFrom);
    base += keepFrom;
  }
  return out;
}

// JPEG slices of an .mjpeg recording, cached per file.
const recordings = new WeakMap<File, Promise<Blob[]>>();
export function mjpegFrames(file: File): Promise<Blob[]> {
  let frames = recordings.get(file);
  if (!frames) {
    frames = mjpegFrameRanges(file).then((ranges) =>
      ranges.map(([s, e]) => file.slice(s, e, "image/jpeg")),
    );
    recordings.set(file, frames);
  }
  return frames;
}

function frameName(file: File, n: number): string {
  const baseName = file.name.replace(/\.[^.]+$/, "").replace(/\s+/g, "_");
  return `${baseName}_frame${String(n).padStart(4, "0")}.jpg`;
}

// Per-frame limits.
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
  // Still over the cap at the lowest quality: half size at quality 0.5.
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
  if ("step" in params) {
    const frames = await mjpegFrames(file);
    const step = Math.max(1, Math.round(params.step));
    const last = Math.min(Math.round(params.end), frames.length - 1);
    const total = extractCount(params);
    const out: File[] = [];
    for (let i = Math.max(0, Math.round(params.start)); i <= last; i += step) {
      out.push(new File([frames[i]], frameName(file, out.length + 1), { type: "image/jpeg" }));
      onProgress(out.length, total);
    }
    return out;
  }
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

    // Long side capped at FRAME_MAX_EDGE before encoding.
    const srcW = video.videoWidth || 1920;
    const srcH = video.videoHeight || 1080;
    const longest = Math.max(srcW, srcH);
    const scale = longest > FRAME_MAX_EDGE ? FRAME_MAX_EDGE / longest : 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(srcW * scale));
    canvas.height = Math.max(1, Math.round(srcH * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");

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
        // A seek to the current time may not fire seeked, so it is nudged.
        video.currentTime = t === video.currentTime ? t + 0.001 : t;
        // Some codecs never fire seeked.
        setTimeout(() => {
          if (!done) {
            done = true;
            video.removeEventListener("seeked", onSeeked);
            resolve();
          }
        }, 3000);
      });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const framed = await encodeFrameUnderCap(canvas, frameName(file, i + 1));
      if (framed) out.push(framed);
      onProgress(i + 1, totalFrames);
    }

    return out;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Per-file limit for videos, which the browser decodes.
export const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

// A recording is scanned in chunks, never decoded, so it may be larger.
export const MAX_RECORDING_BYTES = 2 * 1024 * 1024 * 1024;

// The largest file the import takes for `file`.
export function maxImportBytes(file: File): number {
  return isMjpeg(file) ? MAX_RECORDING_BYTES : MAX_VIDEO_BYTES;
}
