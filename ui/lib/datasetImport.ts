// Client-side parsing for the "import an already-labelled dataset as a
// fully-editable project" flow. Everything here runs in the user's tab:
// we collect the dataset's files (from a folder picker OR a .zip), parse
// the annotations into the canonical editable-box shape the editor +
// exporters consume, and prepare each image (optional downscale) with its
// boxes rescaled to match. The result is fed to the batched ingest endpoint
// (/api/v2/projects/{id}/imports/raw_batch) which writes them as
// editedBoxes (editedBoxesSet=True, labelled=True).
//
// Format support is VOC-first but structured so COCO / YOLO parsers slot
// in as additional `parse*` functions behind `detectFormat`/`parseDataset`.

import { unzip } from "fflate";

// ── Public types ──────────────────────────────────────────────────────────

// One editable box in absolute top-left pixel coords on the image it
// belongs to. Maps 1:1 to a Pascal VOC <bndbox> (xmin/ymin/xmax/ymax) and
// to the backend's editedBox shape. score is null for ground-truth imports.
export type ImportBox = {
  id: string;
  label: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  score: number | null;
  mask?: { polygons: number[][][] } | null;
};

// One image + its annotation, before upload-time preparation. `width`/
// `height` are the annotation's declared dims (VOC <size>); 0 when absent
// (resolved by decoding at prepare time). `boxes` are in that declared
// pixel space.
export type ImportItem = {
  file: File;
  originalName: string;
  boxes: ImportBox[];
  width: number;
  height: number;
};

export type DatasetFormat = "voc" | "coco" | "yolo" | "unknown";

export type ParsedDataset = {
  format: DatasetFormat;
  items: ImportItem[];
  classes: string[]; // unique, lowercased, first-seen order
  stats: {
    images: number;
    annotated: number; // images that had an annotation file
    background: number; // images with zero boxes (explicitly empty or unpaired)
    boxes: number;
    droppedBoxes: number; // degenerate / unparseable objects skipped
    unpairedAnnotations: number; // annotation files with no matching image
  };
  warnings: string[];
};

export type CollectedFile = { path: string; file: File };

// Soft ceiling above which a single in-browser .zip expand risks OOMing the
// tab (fflate decompresses into memory). The folder picker streams Files
// lazily and has no such limit, so we steer large datasets there.
export const ZIP_SOFT_LIMIT_BYTES = 1_500 * 1024 * 1024;

const IMAGE_EXT_RE = /\.(jpe?g|png|webp|gif|bmp|tiff?|heic|heif|avif)$/i;
const JPEG_EXT_RE = /\.jpe?g$/i;
const JPEG_QUALITY = 0.92;

// ── File collection (folder picker OR zip) ─────────────────────────────────

// Normalise whatever the picker hands us into a flat {path, File}[] list.
// A single .zip is expanded client-side; everything else (a webkitdirectory
// FileList, a multi-select, or a folder drop already flattened to Files) is
// passed through, preferring webkitRelativePath so we keep the
// Annotations/ vs JPEGImages/ structure for pairing.
export async function collectInput(files: File[]): Promise<CollectedFile[]> {
  if (files.length === 1 && /\.zip$/i.test(files[0].name)) {
    return collectFromZip(files[0]);
  }
  return files.map((f) => ({ path: relPath(f), file: f }));
}

function relPath(f: File): string {
  const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
  return rel && rel.length > 0 ? rel : f.name;
}

async function collectFromZip(zip: File): Promise<CollectedFile[]> {
  const buf = new Uint8Array(await zip.arrayBuffer());
  const entries = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    // Only inflate the files we care about — images + annotation text —
    // so a zip that also bundles model weights / videos doesn't balloon
    // memory decompressing bytes we'll throw away.
    unzip(
      buf,
      {
        filter: (file) =>
          IMAGE_EXT_RE.test(file.name) || /\.(xml|json|txt|yaml|yml)$/i.test(file.name),
      },
      (err, data) => (err ? reject(err) : resolve(data)),
    );
  });
  const out: CollectedFile[] = [];
  for (const [path, data] of Object.entries(entries)) {
    if (path.endsWith("/") || data.byteLength === 0) continue; // directory entry
    const name = baseName(path);
    const type = mimeForName(name);
    out.push({
      path,
      // `data` is a Uint8Array (a valid BlobPart at runtime); the cast
      // only satisfies the newer Uint8Array<ArrayBufferLike> generic.
      file: new File([data as BlobPart], name, type ? { type } : undefined),
    });
  }
  return out;
}

function mimeForName(name: string): string | undefined {
  if (JPEG_EXT_RE.test(name)) return "image/jpeg";
  if (/\.png$/i.test(name)) return "image/png";
  if (/\.webp$/i.test(name)) return "image/webp";
  if (/\.bmp$/i.test(name)) return "image/bmp";
  if (/\.(tiff?)$/i.test(name)) return "image/tiff";
  if (/\.xml$/i.test(name)) return "application/xml";
  if (/\.json$/i.test(name)) return "application/json";
  return undefined;
}

// ── Format detection + dispatch ────────────────────────────────────────────

export function detectFormat(files: CollectedFile[]): DatasetFormat {
  let hasXml = false;
  let hasCocoJson = false;
  let hasYoloTxt = false;
  let hasYoloMeta = false;
  for (const cf of files) {
    const n = baseName(cf.path).toLowerCase();
    if (n.endsWith(".xml")) hasXml = true;
    else if (n.endsWith(".json")) hasCocoJson = true;
    else if (n.endsWith(".txt") && n !== "classes.txt") hasYoloTxt = true;
    else if (n === "classes.txt" || n === "data.yaml" || n === "data.yml") hasYoloMeta = true;
  }
  if (hasXml) return "voc";
  if (hasCocoJson) return "coco";
  if (hasYoloTxt && hasYoloMeta) return "yolo";
  return "unknown";
}

// Parse a collected dataset. Only Pascal VOC is implemented today; COCO /
// YOLO are detected and rejected with a clear message until their parsers
// land (the rest of the pipeline is format-agnostic).
export async function parseDataset(
  files: CollectedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  const fmt = detectFormat(files);
  if (fmt === "voc") return parseVoc(files, onProgress);
  if (fmt === "coco" || fmt === "yolo") {
    throw new Error(
      `${fmt.toUpperCase()} import isn't supported yet — only Pascal VOC (Annotations/*.xml + images) for now.`,
    );
  }
  throw new Error(
    "Couldn't recognise this as a Pascal VOC dataset (no .xml annotation files found alongside images).",
  );
}

// ── Pascal VOC ─────────────────────────────────────────────────────────────

async function parseVoc(
  files: CollectedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  // Index images by basename stem so an annotation can find its image
  // regardless of which subdir each lives in (Annotations/ vs JPEGImages/).
  const imageByStem = new Map<string, CollectedFile>();
  const imageByName = new Map<string, CollectedFile>();
  const stemCollisions = new Set<string>();
  const xmls: CollectedFile[] = [];
  for (const cf of files) {
    const name = baseName(cf.path);
    if (/\.xml$/i.test(name)) {
      xmls.push(cf);
    } else if (IMAGE_EXT_RE.test(name)) {
      const stem = stemOf(name).toLowerCase();
      // Two images sharing a basename stem (e.g. img.jpg + img.png) make the
      // stem an ambiguous pairing key — record it so stem-only resolution
      // skips it rather than silently pairing an annotation to the wrong file.
      if (imageByStem.has(stem)) stemCollisions.add(stem);
      else imageByStem.set(stem, cf);
      imageByName.set(name.toLowerCase(), cf);
    }
  }

  const warnings: string[] = [];
  const items: ImportItem[] = [];
  const classes: string[] = [];
  const classSeen = new Set<string>();
  const usedStems = new Set<string>();
  let boxesTotal = 0;
  let dropped = 0;
  let unpairedAnnotations = 0;
  let annotated = 0;
  let background = 0;

  const total = xmls.length;
  let done = 0;
  for (const xf of xmls) {
    let parsed: VocXml;
    try {
      parsed = parseVocXml(await xf.file.text());
    } catch {
      unpairedAnnotations++; // unreadable annotation — count + skip
      onProgress?.(++done, total);
      continue;
    }
    // Resolve the image: exact <filename> match first (incl. extension), then
    // an UNAMBIGUOUS stem match (a stem shared by >1 image is skipped so an
    // annotation never pairs to the wrong file/format).
    const byStem = (s: string) => (stemCollisions.has(s) ? undefined : imageByStem.get(s));
    const img =
      (parsed.filename && imageByName.get(parsed.filename.toLowerCase())) ||
      byStem(stemOf(parsed.filename || baseName(xf.path)).toLowerCase()) ||
      byStem(stemOf(baseName(xf.path)).toLowerCase());
    if (!img) {
      unpairedAnnotations++;
      onProgress?.(++done, total);
      continue;
    }
    usedStems.add(stemOf(baseName(img.path)).toLowerCase());

    const boxes: ImportBox[] = [];
    parsed.objects.forEach((o, i) => {
      const label = (o.name || "").trim().toLowerCase();
      if (!label) {
        dropped++;
        return;
      }
      let { xmin: x0, ymin: y0, xmax: x1, ymax: y1 } = o;
      if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) {
        dropped++;
        return;
      }
      if (x1 < x0) [x0, x1] = [x1, x0];
      if (y1 < y0) [y0, y1] = [y1, y0];
      if (x1 - x0 < 1 || y1 - y0 < 1) {
        dropped++;
        return;
      }
      boxes.push({ id: `voc_${i}`, label, x0, y0, x1, y1, score: null });
      if (!classSeen.has(label)) {
        classSeen.add(label);
        classes.push(label);
      }
    });

    boxesTotal += boxes.length;
    annotated++;
    if (boxes.length === 0) background++;
    items.push({
      file: img.file,
      originalName: baseName(img.path),
      boxes,
      width: parsed.width,
      height: parsed.height,
    });
    onProgress?.(++done, total);
  }

  // Images with no annotation file become background (negative) frames —
  // valid, intentionally-empty training data. Surfaced in stats so the
  // user can spot a pairing problem (e.g. a renamed folder).
  let unpairedImages = 0;
  for (const [stem, cf] of imageByStem) {
    if (usedStems.has(stem)) continue;
    unpairedImages++;
    background++;
    items.push({
      file: cf.file,
      originalName: baseName(cf.path),
      boxes: [],
      width: 0,
      height: 0,
    });
  }

  if (unpairedAnnotations > 0)
    warnings.push(`${unpairedAnnotations} annotation file(s) had no matching image and were skipped.`);
  if (unpairedImages > 0)
    warnings.push(`${unpairedImages} image(s) had no annotation and were imported as empty background frames.`);
  if (dropped > 0)
    warnings.push(`${dropped} box(es) were unlabelled or degenerate (zero-area) and were dropped.`);
  if (stemCollisions.size > 0)
    warnings.push(`${stemCollisions.size} image name(s) were ambiguous (same name, different extension); their annotations may not have paired.`);
  if (items.length === 0)
    warnings.push("No image/annotation pairs were found — check the folder layout.");

  return {
    format: "voc",
    items,
    classes,
    stats: {
      images: items.length,
      annotated,
      background,
      boxes: boxesTotal,
      droppedBoxes: dropped,
      unpairedAnnotations,
    },
    warnings,
  };
}

type VocXml = {
  filename: string;
  width: number;
  height: number;
  objects: { name: string; xmin: number; ymin: number; xmax: number; ymax: number }[];
};

function parseVocXml(text: string): VocXml {
  const doc = new DOMParser().parseFromString(text, "application/xml");
  if (doc.querySelector("parsererror")) throw new Error("malformed XML");
  const ann = doc.querySelector("annotation") || doc.documentElement;
  if (!ann) throw new Error("no <annotation> root");
  const size = ann.querySelector("size");
  const objects = Array.from(ann.querySelectorAll("object")).map((o) => {
    const bb = o.querySelector("bndbox");
    return {
      name: textOf(o.querySelector("name")),
      xmin: num(textOf(bb?.querySelector("xmin") ?? null)),
      ymin: num(textOf(bb?.querySelector("ymin") ?? null)),
      xmax: num(textOf(bb?.querySelector("xmax") ?? null)),
      ymax: num(textOf(bb?.querySelector("ymax") ?? null)),
    };
  });
  return {
    filename: textOf(ann.querySelector("filename")),
    width: num(textOf(size?.querySelector("width") ?? null)),
    height: num(textOf(size?.querySelector("height") ?? null)),
    objects,
  };
}

// ── Upload preparation (decode, optional downscale, box rescale) ───────────

// Prepare one item for upload. Decodes the image (createImageBitmap bakes
// EXIF orientation, giving upright pixel dims), maps the annotation's boxes
// from its declared <size> space into true decoded space (a no-op for
// well-formed datasets), then either downscales to `maxSide` (re-encoding a
// high-quality JPEG and scaling boxes by the same factor) or preserves the
// original resolution. Boxes always come out in the EXACT pixel space of the
// returned File, which is what the backend re-decodes — so coordinates stay
// pixel-accurate end to end.
export async function prepareItemForUpload(
  item: ImportItem,
  maxSide: number | null,
): Promise<{ file: File; boxes: ImportBox[]; width: number; height: number }> {
  const { bitmap, trueW, trueH, viaBitmap } = await decodeImage(item.file);
  try {
    let boxes = item.boxes;
    const declaredW = item.width || trueW;
    const declaredH = item.height || trueH;
    const sx = trueW / (declaredW || trueW);
    const sy = trueH / (declaredH || trueH);
    if (sx !== 1 || sy !== 1) boxes = boxes.map((b) => scaleBox(b, sx, sy));

    const longest = Math.max(trueW, trueH);
    if (maxSide && longest > maxSide) {
      const s = maxSide / longest;
      const outW = Math.max(1, Math.round(trueW * s));
      const outH = Math.max(1, Math.round(trueH * s));
      const file = await encodeCanvas(bitmap, outW, outH, item.originalName);
      const fx = outW / trueW;
      const fy = outH / trueH;
      return { file, boxes: boxes.map((b) => scaleBox(b, fx, fy)), width: outW, height: outH };
    }

    // Preserve full resolution. A JPEG decoded via createImageBitmap passes
    // through its original bytes (lossless + fast): the FE box space and the
    // backend's recorded dims both derive from the SAME bytes + EXIF tag, so an
    // orientation tag is applied identically on both sides and they always
    // agree. The <img> fallback can report unreliable oriented dims, so when it
    // was used we re-encode through the canvas to bake an upright, tag-less JPEG
    // whose pixel space provably matches the boxes. Non-JPEG sources are always
    // re-encoded so the stored bytes are uniformly decodable downstream.
    if (viaBitmap && JPEG_EXT_RE.test(item.originalName) && /jpeg/i.test(item.file.type || "")) {
      return { file: await materialise(item.file), boxes, width: trueW, height: trueH };
    }
    const file = await encodeCanvas(bitmap, trueW, trueH, item.originalName);
    return { file, boxes, width: trueW, height: trueH };
  } finally {
    if (typeof (bitmap as ImageBitmap).close === "function") {
      (bitmap as ImageBitmap).close();
    }
  }
}

async function decodeImage(
  file: File,
): Promise<{ bitmap: ImageBitmap | HTMLImageElement; trueW: number; trueH: number; viaBitmap: boolean }> {
  if (typeof createImageBitmap === "function") {
    try {
      const bm = await createImageBitmap(file, { imageOrientation: "from-image" });
      return { bitmap: bm, trueW: bm.width, trueH: bm.height, viaBitmap: true };
    } catch {
      /* fall through to <img> */
    }
  }
  // Fallback path. <img>.naturalWidth/Height can report physical (un-oriented)
  // dims inconsistently across browsers, so callers must not trust these for a
  // lossless byte passthrough (see prepareItemForUpload).
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = url;
    });
    return { bitmap: img, trueW: img.naturalWidth, trueH: img.naturalHeight, viaBitmap: false };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function encodeCanvas(
  bitmap: ImageBitmap | HTMLImageElement,
  w: number,
  h: number,
  sourceName: string,
): Promise<File> {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  ctx.drawImage(bitmap, 0, 0, w, h);
  const blob: Blob | null = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
  );
  if (!blob) throw new Error("image encode failed");
  const name = `${stemOf(sourceName) || "image"}.jpg`;
  return materialise(blob, name);
}

function scaleBox(b: ImportBox, sx: number, sy: number): ImportBox {
  return {
    ...b,
    x0: b.x0 * sx,
    y0: b.y0 * sy,
    x1: b.x1 * sx,
    y1: b.y1 * sy,
    mask: b.mask
      ? { polygons: b.mask.polygons.map((ring) => ring.map(([x, y]) => [x * sx, y * sy])) }
      : b.mask,
  };
}

// Copy bytes into a fresh ArrayBuffer-backed File so the upload survives any
// later GC of the source blob (Safari blob-detach defence; mirrors resize.ts).
async function materialise(blob: Blob, name?: string): Promise<File> {
  const buf = await blob.arrayBuffer();
  const type = blob.type && blob.type.startsWith("image/") ? blob.type : "image/jpeg";
  const fname = name ?? (blob instanceof File ? blob.name : "image.jpg");
  return new File([new Blob([buf], { type })], fname, { type });
}

// ── Batched upload ─────────────────────────────────────────────────────────

export type UploadProgress = { done: number; total: number; failed: number; droppedBoxes: number };

// Upload a parsed dataset to a project via the batched ingest endpoint.
// Items are prepared (decoded + optionally downscaled, boxes rescaled) at
// bounded concurrency, accumulated into batches of `batchSize` (well under
// the backend's 100-file/request cap), and POSTed one batch at a time —
// each request is a single manifest write server-side, so a multi-thousand
// image import stays off the O(n^2) manifest-rewrite cliff. `poster` injects
// the authed fetch (apiFetch) so this stays decoupled from auth/session.
export async function uploadDataset(
  items: ImportItem[],
  opts: { maxSide: number | null; batchSize?: number; prepConcurrency?: number },
  poster: (form: FormData) => Promise<Response>,
  onProgress?: (p: UploadProgress) => void,
): Promise<UploadProgress & { errors: string[] }> {
  const batchSize = Math.max(1, Math.min(opts.batchSize ?? 20, 100));
  const prepConcurrency = Math.max(1, opts.prepConcurrency ?? 4);
  const total = items.length;
  let done = 0;
  let failed = 0;
  let droppedBoxes = 0;
  const errors: string[] = [];

  for (let start = 0; start < items.length; start += batchSize) {
    const batch = items.slice(start, start + batchSize);
    // Prepare the batch at bounded concurrency to cap peak memory (a 4K
    // decode is ~33 MB; we keep only a few in flight at once).
    const prepared: ({ file: File; boxes: ImportBox[]; key: string } | null)[] = new Array(
      batch.length,
    ).fill(null);
    let cursor = 0;
    const prepWorker = async () => {
      for (;;) {
        const i = cursor++;
        if (i >= batch.length) return;
        try {
          const p = await prepareItemForUpload(batch[i], opts.maxSide);
          prepared[i] = { file: p.file, boxes: p.boxes, key: randomKey() };
        } catch (e) {
          failed++;
          errors.push(`${batch[i].originalName}: ${errMsg(e)}`);
        }
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(prepConcurrency, batch.length) }, prepWorker),
    );

    const form = new FormData();
    let n = 0;
    for (const p of prepared) {
      if (!p) continue;
      form.append("images", p.file);
      form.append("boxes", JSON.stringify(p.boxes));
      form.append("idempotency_key", p.key);
      n++;
    }
    if (n > 0) {
      try {
        const r = await poster(form);
        if (!r.ok) throw new Error(`http ${r.status}`);
        const data = (await r.json()) as {
          results?: { status?: string; dropped_boxes?: number }[];
        };
        for (const res of data.results ?? []) {
          if (res.status === "ok") done++;
          else failed++;
          droppedBoxes += res.dropped_boxes ?? 0;
        }
      } catch (e) {
        failed += n;
        errors.push(`batch @${start}: ${errMsg(e)}`);
      }
    }
    onProgress?.({ done, total, failed, droppedBoxes });
  }
  return { done, total, failed, droppedBoxes, errors };
}

function randomKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `imp_${Math.random().toString(36).slice(2)}_${Date.now()}`;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// ── small utils ────────────────────────────────────────────────────────────

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function stemOf(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}

function textOf(el: Element | null): string {
  return (el?.textContent ?? "").trim();
}

function num(s: string): number {
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : NaN;
}
