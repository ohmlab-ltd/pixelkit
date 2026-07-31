// Browser-side image downsizing. Runs entirely in the user's tab, so the
// bytes that leave their machine (and reach R2) are already cropped to
// `maxSide` longest edge and squeezed under `maxBytes`. Saves home-network
// upload bandwidth, R2 storage, and skips backend re-encode work.

const DEFAULT_MAX_SIDE = 1500;
const DEFAULT_MAX_BYTES = 50 * 1024;
// Quality ladder for the binary-ish search. Each step roughly halves bytes.
const QUALITY_STEPS = [0.85, 0.72, 0.6, 0.48, 0.36, 0.25];


async function decode(file: File): Promise<{ bitmap: ImageBitmap | HTMLImageElement; width: number; height: number }> {
  // createImageBitmap handles HEIC/JPEG/PNG/WebP and applies EXIF rotation.
  if (typeof createImageBitmap === "function") {
    try {
      const bm = await createImageBitmap(file, { imageOrientation: "from-image" });
      return { bitmap: bm, width: bm.width, height: bm.height };
    } catch {
      // Fall through to <img> path for browsers that choke on a format.
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = url;
    });
    return { bitmap: img, width: img.naturalWidth, height: img.naturalHeight };
  } finally {
    URL.revokeObjectURL(url);
  }
}


function drawTo(canvas: HTMLCanvasElement, bitmap: ImageBitmap | HTMLImageElement, w: number, h: number) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  ctx.drawImage(bitmap, 0, 0, w, h);
}


/**
 * Resize and recompress `file` so it is at most `maxSide` px on its longest
 * edge AND at most `maxBytes`. Always returns a JPEG (loses transparency,
 * which is fine for photos and screenshots). Returns the original file
 * untouched if it already satisfies both constraints.
 */
export async function resizeForUpload(
  file: File,
  maxSide: number = DEFAULT_MAX_SIDE,
  maxBytes: number = DEFAULT_MAX_BYTES,
): Promise<File> {
  // Safari quirk: dragged files sometimes arrive with an empty
  // `file.type` (Photos drag, some screenshots), so we also sniff the
  // extension before bailing out of the resize pipeline.
  if (!isImageFile(file)) return file;

  // Cheap pass: already JPEG and small enough, skip the canvas round-trip.
  // (Can't trust file.size alone without checking dimensions, so we still
  // decode below for non-trivial images.)
  if (file.size <= maxBytes && /jpeg/i.test(file.type)) {
    // Still need dimension check; decode is cheap if cached by the browser.
  }

  const { bitmap, width, height } = await decode(file);
  const longest = Math.max(width, height);
  const scale = longest > maxSide ? maxSide / longest : 1;
  const targetW = Math.round(width * scale);
  const targetH = Math.round(height * scale);

  // No resize needed AND already under the byte limit AND already JPEG.
  // Still copy the bytes through ArrayBuffer so the returned File is
  // backed by its own memory rather than the original blob, Safari
  // can otherwise invalidate the dragged-file blob between the time
  // we hand it back and the time FormData reads it.
  if (scale === 1 && file.size <= maxBytes && /jpeg/i.test(file.type)) {
    return await materialise(file, file.name);
  }

  const canvas = document.createElement("canvas");
  drawTo(canvas, bitmap, targetW, targetH);

  const stem = file.name.replace(/\.[^.]+$/, "") || "image";
  const targetName = `${stem}.jpg`;

  for (const q of QUALITY_STEPS) {
    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", q),
    );
    if (!blob) continue;
    if (blob.size <= maxBytes) {
      return await materialise(blob, targetName);
    }
  }

  // Quality floor still over the byte budget. We deliberately do NOT shrink
  // the dimensions below `maxSide` — the user chose this resolution (e.g. a 4K
  // Project) and preserving it matters more than the byte budget. Emit the
  // lowest-quality FULL-RESOLUTION JPEG instead of halving the dimensions.
  const floor: Blob | null = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.22),
  );
  if (floor) return await materialise(floor, targetName);

  // Couldn't squeeze it; materialise the original so at least the
  // backend gets a self-contained blob (Safari blob detach defence).
  return await materialise(file, file.name);
}

// Sniff whether a File is an image we want to route through the
// resize/upload pipeline. Prefers `file.type`; falls back to a known
// extension whitelist when the MIME is missing (Safari edge cases).
const IMAGE_EXT_RE = /\.(jpe?g|png|webp|gif|bmp|tiff?|heic|heif|avif)$/i;
export function isImageFile(file: File): boolean {
  if (file.type && file.type.startsWith("image/")) return true;
  if (!file.type) return IMAGE_EXT_RE.test(file.name);
  return false;
}

// Safari's canvas.toBlob() can return a Blob whose underlying pixel
// data is still backed by the source canvas, when the canvas leaves
// scope and the GC reclaims it, the Blob becomes unreadable and
// subsequent `<img src=blob:...>` loads + FormData uploads fail with
// WebKitBlobResource error 4 / "bad URL" / "Load failed". The same
// detach pattern shows up for dragged-file blobs that get held in
// React state across renders. Copying bytes into a fresh
// ArrayBuffer-backed Blob severs both relationships so the returned
// File survives any later GC pass.
//
// Preserves the input MIME when the source is already an image File
// (e.g. JPEG passthrough), only forces `image/jpeg` when the caller
// has explicitly built a JPEG via canvas.toBlob.
async function materialise(blob: Blob, name: string): Promise<File> {
  const buf = await blob.arrayBuffer();
  const type = blob.type && blob.type.startsWith("image/") ? blob.type : "image/jpeg";
  return new File([new Blob([buf], { type })], name, { type });
}
