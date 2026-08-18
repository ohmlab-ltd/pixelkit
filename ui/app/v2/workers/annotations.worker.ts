// Dedicated annotations worker. Owns the three jobs that used to
// pile up on the main thread during heavy gallery / viewer activity:
//
//   1. parseViewport - JSON-parse a /v3/viewport response body and
//      unwrap each WireDetection into the ImportDetection shape the
//      FE actually reads. On a 9000-detection project this is the
//      slowest single piece of mid-flight work (10–30 ms blocking
//      the main thread per batch fetch).
//   2. parseSingle - same, for the legacy per-image /annotations/{id}
//      endpoint. Used by the flag-off path so we get the worker
//      win without changing /v3 vs /v2 wire format.
//   3. labelStats - sum per-id labelStats into a project-wide
//      aggregate. The store's mirror-writes effect runs after every
//      setImports; pushing this off-thread keeps a labelling-job
//      progress poll (which fires setImports at 5 Hz on a 900-image
//      project) from janking the gallery scroll.
//
// Inputs come in as ArrayBuffer transferables (zero-copy) for the
// JSON jobs. Outputs are plain objects (small enough that the
// structured-clone cost is negligible).
//
// msgpack decode runs in the worker too so the main thread doesn't
// pay any parse cost regardless of wire format.
import { unpack as msgpackUnpack } from "msgpackr";

// No imports from the app bundle - the worker is a separate
// compilation unit. The minimal logic the worker needs
// (unwrapWireDetection, stripTransientBoxFlags) is inlined below.

// ─── Local types (kept loose; structurally compatible with the
//                 main-bundle types) ────────────────────────────────

type MaskShape = unknown;

type WireDetection = {
  box: number[];
  mask: MaskShape | null;
  embedding?: number[];
  gd_label?: string | null;
  gd_variant?: string | null;
  gd_score?: number | null;
  vlm_label?: string | null;
  vlm_score?: number | null;
  vlm_ms?: number | null;
  embed_nearest_label?: string | null;
  embed_nearest_sim?: number | null;
  embed_sim_for_label?: number | null;
  embed_margin?: number | null;
  pred_label?: string | null;
  pred_source?: string | null;
  rejected?: boolean;
  reject_reason?: string | null;
  ambiguous?: boolean;
  vlm_action?: string | null;
  embed_sims?: unknown;
  embed_sims_dino?: unknown;
  embed_sims_siglip?: unknown;
  siglip_weight?: number | null;
  crop_jpg_b64?: string | null;
};

type ParsedDetection = {
  box: [number, number, number, number];
  mask: MaskShape | null;
  embedding: number[] | undefined;
  gdLabel: string | null;
  gdVariant: string | null;
  gdScore: number | null;
  vlmLabel: string | null;
  vlmScore: number | null;
  vlmMs: number | null;
  embedNearestLabel: string | null;
  embedSimilarity: number | null;
  embedSimilarityForLabel: number | null;
  embedMargin: number | null;
  predLabel: string | null;
  predSource: string | null;
  rejected: boolean;
  rejectReason: string | null;
  ambiguous: boolean;
  vlmAction: string | null;
  embedSims: unknown;
  embedSimsDino: unknown;
  embedSimsSiglip: unknown;
  siglipWeight: number | null;
  cropDataUrl: string;
};

type WireEditableBox = {
  id?: string;
  label?: string;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  score?: number;
  mask?: MaskShape;
  validation?: unknown;
  // Transient flags the worker strips.
  [k: string]: unknown;
};

type ParsedEditedBox = {
  id?: string;
  label?: string;
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  score?: number;
  mask?: MaskShape;
  validation?: unknown;
};

type ParsedRow = {
  detections: ParsedDetection[];
  editedBoxes: ParsedEditedBox[] | null;
  timings: unknown;
};

type ViewportResponse = {
  imports?: Record<string, {
    detections?: WireDetection[];
    editedBoxes?: WireEditableBox[] | null;
    timings?: unknown;
  }>;
};

type SingleResponse = {
  detections?: WireDetection[];
  editedBoxes?: WireEditableBox[] | null;
  timings?: unknown;
};

// ─── Worker request / response envelopes ─────────────────────────

type ReqEnvelope =
  | { op: "parseViewport"; id: number; buffer: ArrayBuffer }
  | { op: "parseSingle"; id: number; buffer: ArrayBuffer }
  // P6: same shape as parseViewport but the buffer carries a msgpack
  // payload rather than JSON. Decoded with msgpackr in the worker
  // so the main thread never sees the raw bytes.
  | { op: "parseViewportMsgpack"; id: number; buffer: ArrayBuffer }
  | { op: "labelStats"; id: number; slim: { id: string; labelStats?: Record<string, number> }[] };

type RespEnvelope =
  | { id: number; ok: true; result: { imports: Record<string, ParsedRow> } }
  | { id: number; ok: true; result: ParsedRow }
  | { id: number; ok: true; result: Record<string, number> }
  | { id: number; ok: false; error: string };

// ─── Conversion helpers ──────────────────────────────────────────

function unwrapWireDetection(d: WireDetection): ParsedDetection {
  return {
    box: [d.box[0], d.box[1], d.box[2], d.box[3]],
    mask: d.mask,
    embedding: d.embedding,
    gdLabel: d.gd_label ?? null,
    gdVariant: d.gd_variant ?? null,
    gdScore: d.gd_score ?? null,
    vlmLabel: d.vlm_label ?? null,
    vlmScore: d.vlm_score ?? null,
    vlmMs: d.vlm_ms ?? null,
    embedNearestLabel: d.embed_nearest_label ?? null,
    embedSimilarity: d.embed_nearest_sim ?? null,
    embedSimilarityForLabel: d.embed_sim_for_label ?? null,
    embedMargin: d.embed_margin ?? null,
    predLabel: d.pred_label ?? d.gd_label ?? null,
    predSource: d.pred_source ?? (d.gd_label ? "gd" : null),
    rejected: !!d.rejected,
    rejectReason: d.reject_reason ?? null,
    ambiguous: !!d.ambiguous,
    vlmAction: d.vlm_action ?? null,
    embedSims: d.embed_sims ?? null,
    embedSimsDino: d.embed_sims_dino ?? null,
    embedSimsSiglip: d.embed_sims_siglip ?? null,
    siglipWeight: typeof d.siglip_weight === "number" ? d.siglip_weight : null,
    cropDataUrl: d.crop_jpg_b64 ? `data:image/jpeg;base64,${d.crop_jpg_b64}` : "",
  };
}

function stripTransientBoxFlags(box: WireEditableBox): ParsedEditedBox {
  const out: ParsedEditedBox = {
    id: box.id,
    label: box.label,
    x0: box.x0,
    y0: box.y0,
    x1: box.x1,
    y1: box.y1,
    score: box.score,
  };
  if (box.mask !== undefined) out.mask = box.mask;
  if (box.validation !== undefined) out.validation = box.validation;
  return out;
}

function rowFromWire(row: {
  detections?: WireDetection[];
  editedBoxes?: WireEditableBox[] | null;
  timings?: unknown;
}): ParsedRow {
  return {
    detections: (row.detections ?? []).map(unwrapWireDetection),
    editedBoxes: Array.isArray(row.editedBoxes)
      ? row.editedBoxes.map(stripTransientBoxFlags)
      : null,
    timings: row.timings,
  };
}

function decodeJson<T>(buffer: ArrayBuffer): T {
  // TextDecoder is the fastest way to get a string from a buffer in
  // a worker. JSON.parse on the resulting string runs entirely on
  // the worker thread.
  const text = new TextDecoder("utf-8").decode(buffer);
  return JSON.parse(text) as T;
}

function aggregateLabelStats(
  rows: { id: string; labelStats?: Record<string, number> }[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const ls = r.labelStats;
    if (!ls) continue;
    for (const [lab, n] of Object.entries(ls)) {
      if (typeof n !== "number" || n === 0) continue;
      out[lab] = (out[lab] ?? 0) + n;
    }
  }
  return out;
}

// ─── Message handler ─────────────────────────────────────────────

self.addEventListener("message", (e: MessageEvent<ReqEnvelope>) => {
  const msg = e.data;
  try {
    if (msg.op === "parseViewport") {
      const parsed = decodeJson<ViewportResponse>(msg.buffer);
      const importsIn = parsed.imports ?? {};
      const out: Record<string, ParsedRow> = {};
      for (const [id, row] of Object.entries(importsIn)) {
        if (!row) continue;
        out[id] = rowFromWire(row);
      }
      const resp: RespEnvelope = {
        id: msg.id,
        ok: true,
        result: { imports: out },
      };
      (self as unknown as Worker).postMessage(resp);
      return;
    }
    if (msg.op === "parseViewportMsgpack") {
      const parsed = msgpackUnpack(new Uint8Array(msg.buffer)) as ViewportResponse;
      const importsIn = parsed.imports ?? {};
      const out: Record<string, ParsedRow> = {};
      for (const [id, row] of Object.entries(importsIn)) {
        if (!row) continue;
        out[id] = rowFromWire(row);
      }
      const resp: RespEnvelope = {
        id: msg.id,
        ok: true,
        result: { imports: out },
      };
      (self as unknown as Worker).postMessage(resp);
      return;
    }
    if (msg.op === "parseSingle") {
      const parsed = decodeJson<SingleResponse>(msg.buffer);
      const resp: RespEnvelope = {
        id: msg.id,
        ok: true,
        result: rowFromWire(parsed),
      };
      (self as unknown as Worker).postMessage(resp);
      return;
    }
    if (msg.op === "labelStats") {
      const result = aggregateLabelStats(msg.slim);
      const resp: RespEnvelope = { id: msg.id, ok: true, result };
      (self as unknown as Worker).postMessage(resp);
      return;
    }
    // Unknown op
    const resp: RespEnvelope = {
      id: (msg as { id: number }).id,
      ok: false,
      error: `unknown op ${(msg as { op: string }).op}`,
    };
    (self as unknown as Worker).postMessage(resp);
  } catch (err) {
    const resp: RespEnvelope = {
      id: (msg as { id?: number }).id ?? -1,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    (self as unknown as Worker).postMessage(resp);
  }
});

// Sentinel so TS treats this file as a module and so the worker
// has a default export under bundlers that need it.
export {};
