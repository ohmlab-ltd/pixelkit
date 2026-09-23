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
  /** Set for imported polygons: ground truth, which a box move must not
   *  replace by re-running segmentation (as for a hand-painted mask). */
  maskEdited?: boolean;
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
  return files
    .map((f) => ({ path: relPath(f), file: f }))
    .filter(({ path }) => !isMacDebris(path));
}

// Finder's __MACOSX/ folders and ._x resource forks: no image, no labels.
function isMacDebris(path: string): boolean {
  return /(^|\/)__MACOSX\//i.test(path) || baseName(path).startsWith("._");
}

function relPath(f: File): string {
  const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
  return rel && rel.length > 0 ? rel : f.name;
}

async function collectFromZip(zip: File): Promise<CollectedFile[]> {
  const buf = new Uint8Array(await zip.arrayBuffer());
  const entries = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    // Only inflate the files we care about - images + annotation text -
    // so a zip that also bundles model weights / videos doesn't balloon
    // memory decompressing bytes we'll throw away.
    unzip(
      buf,
      {
        filter: (file) =>
          IMAGE_EXT_RE.test(file.name) || /\.(xml|json|txt|yaml|yml|names)$/i.test(file.name),
      },
      (err, data) => (err ? reject(err) : resolve(data)),
    );
  });
  const out: CollectedFile[] = [];
  for (const [path, data] of Object.entries(entries)) {
    if (path.endsWith("/")) continue; // directory entry
    const name = baseName(path);
    // Finder zips carry __MACOSX/._x.jpg resource forks that decode as
    // nothing; skip them. A zero-byte .txt is a legitimate empty YOLO
    // label ("no objects"), so only images need bytes.
    if (isMacDebris(path)) continue;
    if (data.byteLength === 0 && !/\.txt$/i.test(name)) continue;
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

// YOLO sidecar files that name classes rather than label an image.
const YOLO_META_RE = /^(classes\.txt|obj\.names|data\.ya?ml|dataset\.ya?ml)$/i;

// Cheap extension-based guess for UI hints. parseDataset() makes the real
// call: COCO is confirmed by sniffing JSON content (a stray metadata.json
// must not hijack a YOLO folder), and YOLO by whether any .txt actually
// parses as label rows.
export function detectFormat(files: CollectedFile[]): DatasetFormat {
  let hasXml = false;
  let hasJson = false;
  let hasTxt = false;
  for (const cf of files) {
    const n = baseName(cf.path).toLowerCase();
    if (n.endsWith(".xml")) hasXml = true;
    else if (n.endsWith(".json")) hasJson = true;
    else if (n.endsWith(".txt") && !YOLO_META_RE.test(n)) hasTxt = true;
  }
  if (hasXml) return "voc";
  if (hasJson) return "coco";
  if (hasTxt) return "yolo";
  return "unknown";
}

// Parse a collected dataset: Pascal VOC, COCO, or YOLO. The rest of the
// pipeline is format-agnostic - every parser yields the same ImportItem
// shape (absolute-pixel boxes, or unit-space boxes flagged by width =
// height = 1 for YOLO, which prepareItemForUpload scales by the decoded
// dims).
export async function parseDataset(
  files: CollectedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  // An unrelated .xml (a settings or metadata file) must not turn a COCO
  // or YOLO dataset into an empty VOC import.
  if (detectFormat(files) === "voc" && (await hasVocXml(files))) return parseVoc(files, onProgress);
  const cocoJsons = await findCocoJsons(files);
  if (cocoJsons.length > 0) return parseCoco(files, cocoJsons, onProgress);
  if (files.some((cf) => /\.txt$/i.test(baseName(cf.path)))) return parseYolo(files, onProgress);
  throw new Error(
    "Couldn't recognise this dataset. Supported layouts: Pascal VOC (Annotations/*.xml next to images), " +
      "COCO (a JSON with images + annotations), or YOLO (labels/*.txt with classes.txt or data.yaml).",
  );
}

async function hasVocXml(files: CollectedFile[]): Promise<boolean> {
  for (const cf of files) {
    if (/\.xml$/i.test(baseName(cf.path)) && /<annotation[\s>]/i.test(await readText(cf.file))) return true;
  }
  return false;
}

// Text in the encoding the byte-order mark names (some Windows tools save
// VOC XML as UTF-16), UTF-8 otherwise.
async function readText(file: Blob): Promise<string> {
  const b = new Uint8Array(await file.arrayBuffer());
  const enc = b[0] === 0xff && b[1] === 0xfe ? "utf-16le" : b[0] === 0xfe && b[1] === 0xff ? "utf-16be" : "utf-8";
  return new TextDecoder(enc).decode(b);
}

// ── Shared image index (COCO / YOLO pairing) ───────────────────────────────

type ImageIndex = {
  all: CollectedFile[]; // every image, in input order
  byPath: Map<string, CollectedFile>; // lowercased relative path, "/"-normalised
  collisions: Set<string>; // stems shared by >1 image (train/a.jpg + val/a.jpg)
  byStem: (stem: string) => CollectedFile | undefined; // unambiguous stems only
  // Resolve an annotation's file reference ("train/a.jpg", "a.jpg") to
  // the ONE image whose path ends with it. The folder picker prefixes
  // every path with the chosen folder's name ("ds/train/a.jpg"), so an
  // exact match never hits; a basename match with train/ and val/ twins
  // is ambiguous and must not pair (it attached one split's boxes to the
  // other split's file). Ambiguous -> undefined.
  bySuffix: (ref: string) => CollectedFile | undefined;
};

function indexImages(files: CollectedFile[]): ImageIndex {
  const all: CollectedFile[] = [];
  const byPath = new Map<string, CollectedFile>();
  const stems = new Map<string, CollectedFile>();
  const collisions = new Set<string>();
  const bySuffixMap = new Map<string, CollectedFile[]>();
  for (const cf of files) {
    const name = baseName(cf.path);
    if (!IMAGE_EXT_RE.test(name)) continue;
    all.push(cf);
    const stem = stemOf(name).toLowerCase();
    if (stems.has(stem)) collisions.add(stem);
    else stems.set(stem, cf);
    const p = normPath(cf.path);
    byPath.set(p, cf);
    const parts = p.split("/");
    for (let i = 0; i < parts.length; i++) {
      const suffix = parts.slice(i).join("/");
      const list = bySuffixMap.get(suffix);
      if (list) list.push(cf);
      else bySuffixMap.set(suffix, [cf]);
    }
  }
  return {
    all,
    byPath,
    collisions,
    byStem: (s) => (collisions.has(s) ? undefined : stems.get(s)),
    bySuffix: (ref) => {
      const list = bySuffixMap.get(normPath(ref));
      return list && list.length === 1 ? list[0] : undefined;
    },
  };
}

function normPath(p: string): string {
  return p.replace(/\\/g, "/").replace(/^\.\//, "").toLowerCase();
}

// The folder of a path, normalised ("" at the top).
function dirOf(p: string): string {
  const n = normPath(p);
  const i = n.lastIndexOf("/");
  return i < 0 ? "" : n.slice(0, i);
}

function joinPath(dir: string, name: string): string {
  return dir ? `${dir}/${name}` : name;
}

// ── COCO ───────────────────────────────────────────────────────────────────

type CocoAnn = {
  id?: number | string;
  image_id: number | string;
  category_id?: number | string;
  bbox?: unknown;
  segmentation?: unknown;
  iscrowd?: number;
};
type CocoJson = {
  images: { id: number | string; file_name?: string; width?: number; height?: number }[];
  annotations: CocoAnn[];
  categories?: { id: number | string; name?: string }[];
};

function isCocoJson(d: unknown): d is CocoJson {
  if (!d || typeof d !== "object") return false;
  const c = d as CocoJson & { categories?: unknown };
  if (!Array.isArray(c.images) || !Array.isArray(c.annotations) || !Array.isArray(c.categories)) return false;
  // Captions files list the same images with no categories and no boxes;
  // taken as COCO, the one that sorted first left every image with none.
  return c.annotations.length === 0 || c.annotations.some(
    (a) => !!a && (Array.isArray((a as { bbox?: unknown }).bbox) || !!(a as { segmentation?: unknown }).segmentation),
  );
}

// Every .json whose content is COCO-shaped (train + val splits commonly
// arrive together; all are merged).
async function findCocoJsons(files: CollectedFile[]): Promise<{ cf: CollectedFile; data: CocoJson }[]> {
  const out: { cf: CollectedFile; data: CocoJson }[] = [];
  for (const cf of files) {
    if (!/\.json$/i.test(baseName(cf.path))) continue;
    let data: unknown;
    try {
      data = JSON.parse(await cf.file.text());
    } catch {
      continue;
    }
    if (isCocoJson(data)) out.push({ cf, data });
  }
  return out;
}

// COCO segmentation -> polygon rings as [x, y] pairs. Polygon form is a list
// of flat [x0, y0, x1, y1, ...] rings; RLE (an object) is skipped, since the
// editor's mask type is polygon-only. Returns "rle" so the caller can count.
function cocoPolygons(seg: unknown): number[][][] | "rle" | null {
  if (!seg) return null;
  if (!Array.isArray(seg)) return typeof seg === "object" ? "rle" : null;
  const rings: number[][][] = [];
  const flatRings: unknown[] = seg.length > 0 && !Array.isArray(seg[0]) ? [seg] : seg;
  for (const ring of flatRings) {
    if (!Array.isArray(ring)) continue;
    const nums = ring.map(Number);
    if (nums.length < 6 || nums.length % 2 !== 0 || !nums.every(Number.isFinite)) continue;
    const pts: number[][] = [];
    for (let i = 0; i < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
    rings.push(pts);
  }
  return rings.length ? rings : null;
}

async function parseCoco(
  files: CollectedFile[],
  jsons: { cf: CollectedFile; data: CocoJson }[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  const idx = indexImages(files);
  const warnings: string[] = [];
  const items: ImportItem[] = [];
  const classes: string[] = [];
  const classSeen = new Set<string>();
  const seenImages = new Set<string>(); // normalised paths already imported
  let boxesTotal = 0;
  let dropped = 0;
  let unpairedAnnotations = 0;
  let annotated = 0;
  let background = 0;
  let duplicates = 0;
  let rleMasks = 0;
  let crowd = 0;

  const addClass = (label: string) => {
    if (!classSeen.has(label)) {
      classSeen.add(label);
      classes.push(label);
    }
  };

  const total = jsons.reduce((n, j) => n + j.data.images.length, 0);
  let done = 0;
  for (const { data } of jsons) {
    // Declared categories seed the class list in declared order, so a
    // label set survives even when a split has no instance of a class.
    const catName = new Map<string, string>();
    for (const c of data.categories ?? []) {
      if (!c || c.id == null) continue;
      const name = String(c.name ?? c.id).trim().toLowerCase();
      if (!name) continue;
      catName.set(String(c.id), name);
      addClass(name);
    }
    const annsByImage = new Map<string, CocoAnn[]>();
    for (const a of data.annotations) {
      if (!a || a.image_id == null) continue;
      const k = String(a.image_id);
      const list = annsByImage.get(k);
      if (list) list.push(a);
      else annsByImage.set(k, [a]);
    }

    for (const im of data.images) {
      onProgress?.(++done, total);
      if (!im || im.id == null) {
        unpairedAnnotations++;
        continue;
      }
      const fileName = String(im.file_name ?? "");
      const base = baseName(fileName).toLowerCase();
      const img =
        (base && idx.bySuffix(fileName)) ||
        (base && !fileName.includes("/") && !fileName.includes("\\") ? idx.byStem(stemOf(base)) : undefined);
      if (!img) {
        unpairedAnnotations++;
        continue;
      }
      const key = normPath(img.path);
      if (seenImages.has(key)) {
        duplicates++;
        continue;
      }
      seenImages.add(key);

      const boxes: ImportBox[] = [];
      (annsByImage.get(String(im.id)) ?? []).forEach((a, i) => {
        // iscrowd=1 marks a crowd region (one outline around many people),
        // not an object; converters such as Ultralytics' skip them.
        if (a.iscrowd === 1) {
          crowd++;
          return;
        }
        const label =
          catName.get(String(a.category_id)) ??
          (a.category_id != null ? String(a.category_id).trim().toLowerCase() : "");
        if (!label) {
          dropped++;
          return;
        }
        const polys = cocoPolygons(a.segmentation);
        if (polys === "rle") rleMasks++;
        const bb = Array.isArray(a.bbox) ? a.bbox.map(Number) : [];
        let x0: number, y0: number, x1: number, y1: number;
        if (bb.length === 4 && bb.every(Number.isFinite)) {
          [x0, y0] = [bb[0], bb[1]];
          [x1, y1] = [bb[0] + bb[2], bb[1] + bb[3]];
        } else if (polys && polys !== "rle") {
          const xs = polys.flat().map((p) => p[0]);
          const ys = polys.flat().map((p) => p[1]);
          [x0, x1] = [Math.min(...xs), Math.max(...xs)];
          [y0, y1] = [Math.min(...ys), Math.max(...ys)];
        } else {
          dropped++;
          return;
        }
        if (x1 < x0) [x0, x1] = [x1, x0];
        if (y1 < y0) [y0, y1] = [y1, y0];
        if (x1 - x0 < 1 || y1 - y0 < 1) {
          dropped++;
          return;
        }
        boxes.push({
          // Index keeps ids unique even when some annotations carry an
          // id and others do not (a.id 1 vs index 1 collided).
          id: a.id != null ? `coco_${a.id}_${i}` : `coco_${i}`,
          label,
          x0,
          y0,
          x1,
          y1,
          score: null,
          mask: polys && polys !== "rle" ? { polygons: polys } : null,
          ...(polys && polys !== "rle" ? { maskEdited: true } : {}),
        });
        addClass(label);
      });

      boxesTotal += boxes.length;
      annotated++;
      if (boxes.length === 0) background++;
      items.push({
        file: img.file,
        originalName: baseName(img.path),
        boxes,
        width: Number.isFinite(Number(im.width)) ? Number(im.width) : 0,
        height: Number.isFinite(Number(im.height)) ? Number(im.height) : 0,
      });
    }
  }

  let unpairedImages = 0;
  for (const cf of idx.all) {
    if (seenImages.has(normPath(cf.path))) continue;
    unpairedImages++;
    background++;
    items.push({ file: cf.file, originalName: baseName(cf.path), boxes: [], width: 0, height: 0 });
  }

  if (unpairedAnnotations > 0)
    warnings.push(`${unpairedAnnotations} image entr${unpairedAnnotations === 1 ? "y" : "ies"} in the JSON had no matching file and were skipped.`);
  if (unpairedImages > 0)
    warnings.push(`${unpairedImages} image(s) were not referenced by any JSON and were imported as empty background frames.`);
  if (duplicates > 0)
    warnings.push(`${duplicates} image(s) appeared in more than one JSON; the first occurrence was kept.`);
  if (dropped > 0)
    warnings.push(`${dropped} annotation(s) were unlabelled or degenerate (zero-area) and were dropped.`);
  if (rleMasks > 0)
    warnings.push(`${rleMasks} RLE-encoded mask(s) were imported as boxes only (polygon masks are kept).`);
  if (crowd > 0)
    warnings.push(`${crowd} crowd region(s) (iscrowd) were skipped: they outline a group, not one object.`);
  if (idx.collisions.size > 0)
    warnings.push(`${idx.collisions.size} image name(s) were ambiguous (same name, different extension); their annotations may not have paired.`);
  if (items.length === 0) warnings.push("No image/annotation pairs were found - check the folder layout.");

  return {
    format: "coco",
    items,
    classes,
    stats: { images: items.length, annotated, background, boxes: boxesTotal, droppedBoxes: dropped, unpairedAnnotations },
    warnings,
  };
}

// ── YOLO ───────────────────────────────────────────────────────────────────

type YoloRow = {
  cls: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  polygon: number[][] | null;
  score: number | null;
};

// Parse one label file's text. Rows are `cls cx cy w h` (box), the same
// with a trailing confidence (`yolo predict save_txt save_conf`), or
// `cls x1 y1 x2 y2 ...` (YOLO-seg / OBB polygon, >= 3 points), all in
// [0,1] unit space. Returns null when the text is not label rows at all
// (a README.txt, a licence) so the caller can ignore it; an EMPTY file is
// a valid "no objects" label, and a file of malformed rows is a label
// file with dropped rows, not prose.
export function parseYoloLabelText(text: string): { rows: YoloRow[]; bad: number } | null {
  const rows: YoloRow[] = [];
  let bad = 0;
  let lines = 0;
  let looksLikeLabels = false;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    lines++;
    if (/^\d+(\s|,|$)/.test(line)) looksLikeLabels = true;
    const nums = line.split(/[\s,]+/).map(Number);
    if (nums.length < 5 || !nums.every(Number.isFinite) || nums[0] < 0 || !Number.isInteger(nums[0])) {
      bad++;
      continue;
    }
    const cls = nums[0];
    if (nums.length === 5 || nums.length === 6) {
      const [cx, cy, w, h] = nums.slice(1);
      rows.push({
        cls,
        x0: cx - w / 2,
        y0: cy - h / 2,
        x1: cx + w / 2,
        y1: cy + h / 2,
        polygon: null,
        score: nums.length === 6 ? nums[5] : null,
      });
      continue;
    }
    const coords = nums.slice(1);
    if (coords.length % 2 !== 0 || coords.length < 6) {
      bad++;
      continue;
    }
    const pts: number[][] = [];
    for (let i = 0; i < coords.length; i += 2) pts.push([coords[i], coords[i + 1]]);
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    rows.push({
      cls,
      x0: Math.min(...xs),
      y0: Math.min(...ys),
      x1: Math.max(...xs),
      y1: Math.max(...ys),
      polygon: pts,
      score: null,
    });
  }
  if (lines > 0 && rows.length === 0 && !looksLikeLabels) return null;
  return { rows, bad };
}

// Class names from data.yaml / dataset.yaml. Handles every shape seen in
// the wild: inline `names: [a, b]` (possibly spanning lines, possibly
// with a trailing comment), a flow mapping `names: {0: a, 1: b}`, a
// block list (indented, or at column 0 as PyYAML dumps it), and a block
// index mapping with plain or quoted keys.
export function parseYamlNames(text: string): string[] {
  const stripComment = (l: string) => l.replace(/(^|\s)#.*$/, "");
  const lines = text.split(/\r?\n/).map(stripComment);
  const start = lines.findIndex((l) => /^\s*names\s*:/.test(l));
  if (start < 0) return [];
  const clean = (s: string) => s.trim().replace(/^['"]|['"]$/g, "").trim().toLowerCase();
  const namesIndent = /^(\s*)/.exec(lines[start])![1].length;
  const head = lines[start].replace(/^\s*names\s*:/, "").trim();

  const flow = (open: string, close: string): string | null => {
    if (!head.startsWith(open)) return null;
    let body = head;
    let i = start + 1;
    while (!body.includes(close) && i < lines.length) body += " " + lines[i++].trim();
    return body.slice(1, body.lastIndexOf(close) >= 0 ? body.lastIndexOf(close) : undefined);
  };
  const flowList = flow("[", "]");
  if (flowList !== null) return flowList.split(",").map(clean).filter(Boolean);
  const flowMap = flow("{", "}");
  if (flowMap !== null) {
    const mapped: [number, string][] = [];
    for (const kv of flowMap.split(",")) {
      const m = /^\s*['"]?(\d+)['"]?\s*:\s*(.+)$/.exec(kv);
      if (m) mapped.push([Number(m[1]), clean(m[2])]);
    }
    return orderMapped(mapped);
  }

  const listed: string[] = [];
  const mapped: [number, string][] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) continue;
    const indent = /^(\s*)/.exec(l)![1].length;
    const li = /^\s*-\s*(.+)$/.exec(l);
    if (li) {
      if (indent < namesIndent) break;
      listed.push(clean(li[1]));
      continue;
    }
    const mi = /^\s*['"]?(\d+)['"]?\s*:\s*(.+)$/.exec(l);
    if (mi && indent > namesIndent) {
      mapped.push([Number(mi[1]), clean(mi[2])]);
      continue;
    }
    if (indent <= namesIndent) break; // next top-level key ends the block
  }
  if (listed.length) return listed.filter(Boolean);
  return orderMapped(mapped);
}

function orderMapped(mapped: [number, string][]): string[] {
  mapped.sort((a, b) => a[0] - b[0]);
  const out: string[] = [];
  for (const [i, n] of mapped) out[i] = n;
  return Array.from(out, (n) => n ?? "");
}

async function parseYolo(
  files: CollectedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  const idx = indexImages(files);
  const warnings: string[] = [];

  // Class names: classes.txt / obj.names (one per line, index = class id),
  // else data.yaml. Without either, ids become `class_N` so the import
  // still works and the user renames in the ontology panel.
  let names: string[] = [];
  let namesSource = "";
  for (const cf of files) {
    const n = baseName(cf.path).toLowerCase();
    if (n === "classes.txt" || n === "obj.names") {
      // A line's position is its class id: a blank line in the middle is
      // an unnamed class, not a gap to close (that shifted every later id).
      const lines = (await cf.file.text()).split(/\r?\n/).map((s) => s.trim().toLowerCase());
      while (lines.length && !lines[lines.length - 1]) lines.pop();
      names = lines.map((s, i) => s || `class_${i}`);
      if (names.length) {
        namesSource = n;
        break;
      }
    }
  }
  if (!names.length) {
    for (const cf of files) {
      const n = baseName(cf.path).toLowerCase();
      if (/^(data|dataset)\.ya?ml$/.test(n)) {
        names = parseYamlNames(await cf.file.text());
        if (names.length) {
          namesSource = n;
          break;
        }
      }
    }
  }
  if (!names.length)
    warnings.push("No classes.txt or data.yaml found; classes were named class_0, class_1, ... - rename them after import.");
  const nameFor = (id: number) => names[id] || `class_${id}`;

  const labelFiles = files.filter(
    (cf) => /\.txt$/i.test(baseName(cf.path)) && !YOLO_META_RE.test(baseName(cf.path)),
  );
  const items: ImportItem[] = [];
  const classes: string[] = [];
  const classSeen = new Set<string>();
  const usedImages = new Set<string>(); // normalised paths with a label file
  let boxesTotal = 0;
  let dropped = 0;
  let unpairedAnnotations = 0;
  let annotated = 0;
  let background = 0;
  let notLabels = 0;

  // Pair a label to its image: the mirrored path first (labels/x/a.txt ->
  // images/x/a.<ext>, which keeps train/ and val/ twins apart), then an
  // unambiguous stem.
  const imageFor = (labelPath: string): CollectedFile | undefined => {
    const p = normPath(labelPath);
    const mirrored = p.replace(/(^|\/)labels(\/|$)/, "$1images$2").replace(/\.txt$/, "");
    for (const ext of ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "heic", "heif", "avif"]) {
      const hit = idx.byPath.get(`${mirrored}.${ext}`);
      if (hit) return hit;
    }
    return idx.byStem(stemOf(baseName(labelPath)).toLowerCase());
  };

  const total = labelFiles.length;
  let done = 0;
  for (const lf of labelFiles) {
    const parsed = parseYoloLabelText(await lf.file.text());
    onProgress?.(++done, total);
    if (!parsed) {
      notLabels++;
      continue;
    }
    const img = imageFor(lf.path);
    if (!img) {
      unpairedAnnotations++;
      continue;
    }
    const imgKey = normPath(img.path);
    if (usedImages.has(imgKey)) continue; // two label files for one image: first wins
    usedImages.add(imgKey);

    const boxes: ImportBox[] = [];
    dropped += parsed.bad;
    parsed.rows.forEach((r, i) => {
      const x0 = Math.min(Math.max(r.x0, 0), 1);
      const y0 = Math.min(Math.max(r.y0, 0), 1);
      const x1 = Math.min(Math.max(r.x1, 0), 1);
      const y1 = Math.min(Math.max(r.y1, 0), 1);
      if (x1 - x0 <= 0 || y1 - y0 <= 0) {
        dropped++;
        return;
      }
      const label = nameFor(r.cls);
      boxes.push({
        id: `yolo_${i}`,
        label,
        x0,
        y0,
        x1,
        y1,
        score: r.score,
        mask: r.polygon ? { polygons: [r.polygon] } : null,
        ...(r.polygon ? { maskEdited: true } : {}),
      });
      if (!classSeen.has(label)) {
        classSeen.add(label);
        classes.push(label);
      }
    });
    boxesTotal += boxes.length;
    annotated++;
    if (boxes.length === 0) background++;
    // width = height = 1 marks UNIT space: prepareItemForUpload scales by
    // trueW / 1 = trueW, landing the boxes in decoded pixels.
    items.push({ file: img.file, originalName: baseName(img.path), boxes, width: 1, height: 1 });
  }

  if (annotated === 0) {
    throw new Error(
      "No YOLO label files parsed (expected lines like `0 0.5 0.5 0.2 0.3` in labels/*.txt next to images).",
    );
  }

  // Images without a label file: YOLO's own convention for "no objects".
  let unlabelledImages = 0;
  for (const cf of idx.all) {
    if (usedImages.has(normPath(cf.path))) continue;
    unlabelledImages++;
    background++;
    items.push({ file: cf.file, originalName: baseName(cf.path), boxes: [], width: 0, height: 0 });
  }

  // Keep declared class order for the label set when we have names.
  const orderedClasses = names.length
    ? Array.from(new Set([...names.filter(Boolean), ...classes]))
    : classes;

  if (namesSource) warnings.push(`Class names read from ${namesSource}.`);
  if (unpairedAnnotations > 0)
    warnings.push(`${unpairedAnnotations} label file(s) had no matching image and were skipped.`);
  if (unlabelledImages > 0)
    warnings.push(`${unlabelledImages} image(s) had no label file and were imported as empty background frames.`);
  if (notLabels > 0) warnings.push(`${notLabels} .txt file(s) were not label rows and were ignored.`);
  if (dropped > 0) warnings.push(`${dropped} row(s) were malformed or degenerate and were dropped.`);
  if (idx.collisions.size > 0)
    warnings.push(`${idx.collisions.size} image name(s) were ambiguous (same name, different extension); their labels may not have paired.`);

  return {
    format: "yolo",
    items,
    classes: orderedClasses,
    stats: { images: items.length, annotated, background, boxes: boxesTotal, droppedBoxes: dropped, unpairedAnnotations },
    warnings,
  };
}

// ── Pascal VOC ─────────────────────────────────────────────────────────────

async function parseVoc(
  files: CollectedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<ParsedDataset> {
  const images: CollectedFile[] = [];
  const xmls: CollectedFile[] = [];
  for (const cf of files) {
    const name = baseName(cf.path);
    if (/\.xml$/i.test(name)) xmls.push(cf);
    else if (IMAGE_EXT_RE.test(name)) images.push(cf);
  }
  const byPath = new Map(images.map((cf) => [normPath(cf.path), cf]));
  const group = (key: (cf: CollectedFile) => string) => {
    const m = new Map<string, CollectedFile[]>();
    for (const cf of images) {
      const k = key(cf);
      const list = m.get(k);
      if (list) list.push(cf);
      else m.set(k, [cf]);
    }
    return m;
  };
  const byName = group((cf) => baseName(cf.path).toLowerCase());
  const byStem = group((cf) => stemOf(baseName(cf.path)).toLowerCase());
  const only = (list?: CollectedFile[]) => (list && list.length === 1 ? list[0] : undefined);
  // An annotation's picture: beside it, or in the image folder next to its
  // Annotations folder (the VOC layout), by <filename> and then by the
  // XML's own name; anywhere else only when a single image has that name.
  // Twins in train/ and val/ pair within their own split, and a name
  // shared by several images (a.jpg + a.png, split twins) is never guessed.
  const resolveImage = (xmlPath: string, filename: string): CollectedFile | undefined => {
    const dir = dirOf(xmlPath);
    const parent = dirOf(dir);
    const dirs = [dir, ...["jpegimages", "images"].map((d) => joinPath(parent, d))];
    const stem = stemOf(filename || baseName(xmlPath)).toLowerCase();
    if (filename) {
      for (const d of dirs) {
        const hit = byPath.get(normPath(joinPath(d, filename)));
        if (hit) return hit;
      }
    }
    for (const d of dirs) {
      const here = (byStem.get(stem) ?? []).filter((cf) => dirOf(cf.path) === d);
      if (here.length === 1) return here[0];
    }
    return (filename ? only(byName.get(filename.toLowerCase())) : undefined) ?? only(byStem.get(stem));
  };

  const warnings: string[] = [];
  const items: ImportItem[] = [];
  const classes: string[] = [];
  const classSeen = new Set<string>();
  const used = new Set<string>();
  let boxesTotal = 0;
  let dropped = 0;
  let unpairedAnnotations = 0;
  let unreadable = 0;
  let annotated = 0;
  let background = 0;

  const total = xmls.length;
  let done = 0;
  for (const xf of xmls) {
    let parsed: VocXml;
    try {
      parsed = parseVocXml(await readText(xf.file));
    } catch {
      // Unreadable: its picture is left out rather than imported as an
      // empty (negative) frame it is not.
      const img = resolveImage(xf.path, "");
      if (img) used.add(normPath(img.path));
      unreadable++;
      onProgress?.(++done, total);
      continue;
    }
    const img = resolveImage(xf.path, parsed.filename);
    if (!img || used.has(normPath(img.path))) {
      unpairedAnnotations++;
      onProgress?.(++done, total);
      continue;
    }
    used.add(normPath(img.path));

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

  // Images with no annotation file become background (negative) frames -
  // valid, intentionally-empty training data. Surfaced in stats so the
  // user can spot a pairing problem (e.g. a renamed folder).
  let unpairedImages = 0;
  for (const cf of images) {
    if (used.has(normPath(cf.path))) continue;
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
  if (unreadable > 0)
    warnings.push(`${unreadable} annotation file(s) could not be read; their images were left out.`);
  if (unpairedImages > 0)
    warnings.push(`${unpairedImages} image(s) had no annotation and were imported as empty background frames.`);
  if (dropped > 0)
    warnings.push(`${dropped} box(es) were unlabelled or degenerate (zero-area) and were dropped.`);
  if (items.length === 0)
    warnings.push("No image/annotation pairs were found - check the folder layout.");

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
// returned File, which is what the backend re-decodes - so coordinates stay
// pixel-accurate end to end.
export async function prepareItemForUpload(
  item: ImportItem,
  maxSide: number | null,
): Promise<{ file: File; boxes: ImportBox[]; width: number; height: number }> {
  const { bitmap, trueW, trueH, viaBitmap } = await decodeImage(item.file);
  try {
    let boxes = item.boxes;
    let declaredW = item.width || trueW;
    let declaredH = item.height || trueH;
    // Annotated on the stored frame of a photo shown rotated (labelImg and
    // PIL report that size): the declared size is the decoded one turned a
    // quarter, and the boxes must turn with the picture, not stretch to it.
    if (declaredW !== declaredH && declaredW === trueH && declaredH === trueW) {
      const o = await jpegOrientation(item.file);
      if (o >= 5 && o <= 8) {
        boxes = boxes.map((b) => orientBox(b, o, declaredW, declaredH));
        [declaredW, declaredH] = [declaredH, declaredW];
      }
    }
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

// EXIF orientation (1-8) of a JPEG, 1 when it has none. Reads the header
// segments only.
export async function jpegOrientation(file: Blob): Promise<number> {
  const v = new DataView(await file.slice(0, 131072).arrayBuffer());
  if (v.byteLength < 4 || v.getUint16(0) !== 0xffd8) return 1;
  let off = 2;
  while (off + 4 <= v.byteLength) {
    const marker = v.getUint16(off);
    if ((marker & 0xff00) !== 0xff00 || marker === 0xffda) return 1; // image data: no EXIF
    const len = v.getUint16(off + 2);
    if (marker === 0xffe1 && off + 18 <= v.byteLength && v.getUint32(off + 4) === 0x45786966) {
      const tiff = off + 10;
      const le = v.getUint16(tiff) === 0x4949;
      const ifd = tiff + v.getUint32(tiff + 4, le);
      if (ifd + 2 > v.byteLength) return 1;
      for (let i = 0, n = v.getUint16(ifd, le); i < n; i++) {
        const e = ifd + 2 + i * 12;
        if (e + 12 > v.byteLength) return 1;
        if (v.getUint16(e, le) === 0x0112) {
          const o = v.getUint16(e + 8, le);
          return o >= 1 && o <= 8 ? o : 1;
        }
      }
      return 1;
    }
    off += 2 + len;
  }
  return 1;
}

// A box (and its mask) from the stored w x h frame into the upright frame
// of EXIF orientation o (5-8, the quarter turns).
function orientBox(b: ImportBox, o: number, w: number, h: number): ImportBox {
  const turn = ([x, y]: number[]): number[] =>
    o === 5 ? [y, x] : o === 6 ? [h - y, x] : o === 7 ? [h - y, w - x] : [y, w - x];
  const [ax, ay] = turn([b.x0, b.y0]);
  const [bx, by] = turn([b.x1, b.y1]);
  return {
    ...b,
    x0: Math.min(ax, bx),
    y0: Math.min(ay, by),
    x1: Math.max(ax, bx),
    y1: Math.max(ay, by),
    mask: b.mask ? { polygons: b.mask.polygons.map((ring) => ring.map((pt) => turn(pt))) } : b.mask,
  };
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
// the backend's 100-file/request cap), and POSTed one batch at a time -
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
