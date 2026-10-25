import { describe, expect, it } from "vitest";

import {
  extractCount,
  extractVideoFrames,
  isMjpeg,
  isVideoFile,
  MAX_RECORDING_BYTES,
  MAX_VIDEO_BYTES,
  maxImportBytes,
  mjpegFrameRanges,
  mjpegFrames,
} from "./videoFrames";

// A marker segment: FF, marker, 16-bit length including itself, payload.
const seg = (marker: number, payload: number[]) =>
  [0xff, marker, (payload.length + 2) >> 8, (payload.length + 2) & 0xff, ...payload];

// A minimal JPEG: SOI, APP0, extra segments, scan header, entropy-coded bytes, EOI.
const jpeg = (scan: number[], extra: number[][] = []) => [
  0xff, 0xd8,
  ...seg(0xe0, [0x4a, 0x46, 0x49, 0x46, 0]),
  ...extra.flat(),
  ...seg(0xda, [1, 1, 0, 0, 0x3f, 0]),
  ...scan,
  0xff, 0xd9,
];

const bytes = (...parts: number[][]) => new Uint8Array(parts.flat());

// The frame ranges of b, read in one chunk.
const ranges = (b: BlobPart) => mjpegFrameRanges(new Blob([b]));

describe("mjpegFrameRanges", () => {
  it("splits frames laid end to end", async () => {
    const a = jpeg([1, 2, 3]);
    const b = jpeg([4, 5]);
    const c = jpeg([6]);
    expect(await ranges(bytes(a, b, c))).toEqual([
      [0, a.length],
      [a.length, a.length + b.length],
      [a.length + b.length, a.length + b.length + c.length],
    ]);
  });

  it("is not cut short by a thumbnail inside a frame or by look-alike bytes in the scan", async () => {
    // An EXIF segment carrying its own small JPEG, SOI to EOI.
    const exif = seg(0xe1, [0x45, 0x78, 0x69, 0x66, 0, 0, ...jpeg([9, 9])]);
    // Stuffed 0xFF 0x00, a restart marker, and fill bytes before EOI.
    const frame = [...jpeg([0xff, 0x00, 7, 0xff, 0xd3, 8], [exif]).slice(0, -2), 0xff, 0xff, 0xd9];
    const next = jpeg([1]);
    expect(await ranges(bytes(frame, next))).toEqual([
      [0, frame.length],
      [frame.length, frame.length + next.length],
    ]);
  });

  it("skips damaged data and a frame cut off at the end", async () => {
    const a = jpeg([1]);
    const junk = [0x00, 0x13, 0xff, 0x37];
    const b = jpeg([2]);
    const cut = jpeg([3, 4, 5]).slice(0, -4);
    expect(await ranges(bytes(junk, a, junk, b, cut))).toEqual([
      [junk.length, junk.length + a.length],
      [2 * junk.length + a.length, 2 * junk.length + a.length + b.length],
    ]);
    expect(await ranges(new Uint8Array([0xff, 0xd8]))).toEqual([]);
    expect(await ranges(new Uint8Array())).toEqual([]);
  });
});

describe("recordings", () => {
  const rec = (name: string, n: number) =>
    new File([bytes(...Array.from({ length: n }, (_, i) => jpeg([i + 1])))], name);

  it("are recognised by name, whatever type the system gives them", () => {
    expect(isMjpeg(new File([], "recording_20260101_120000.mjpeg"))).toBe(true);
    expect(isMjpeg(new File([], "A.MJPG"))).toBe(true);
    expect(isMjpeg(new File([], "photo.jpeg"))).toBe(false);
    expect(isVideoFile(new File([], "clip.mp4", { type: "video/mp4" }))).toBe(true);
    expect(isVideoFile(new File([], "rec.mjpeg"))).toBe(true);
    expect(isVideoFile(new File([], "photo.jpg", { type: "image/jpeg" }))).toBe(false);
  });

  it("yield their frames as JPEGs, trimmed and sampled", async () => {
    const file = rec("my rec.mjpeg", 10);
    const frames = await mjpegFrames(file);
    expect(frames).toHaveLength(10);
    const first = new Uint8Array(await frames[0].arrayBuffer());
    expect(Array.from(first.slice(0, 2))).toEqual([0xff, 0xd8]);
    expect(Array.from(first.slice(-2))).toEqual([0xff, 0xd9]);

    const params = { start: 2, end: 9, step: 3 };
    expect(extractCount(params)).toBe(3);
    const seen: number[] = [];
    const out = await extractVideoFrames(file, params, (i) => seen.push(i));
    expect(out.map((f) => f.name)).toEqual(["my_rec_frame0001.jpg", "my_rec_frame0002.jpg", "my_rec_frame0003.jpg"]);
    expect(out.every((f) => f.type === "image/jpeg")).toBe(true);
    expect(seen).toEqual([1, 2, 3]);
    // Frames 3, 6 and 9: each scan byte is its frame number.
    const scanByte = async (f: Blob) => new Uint8Array(await f.arrayBuffer())[21];
    expect(await Promise.all(out.map(scanByte))).toEqual([3, 6, 9]);

    // Past the end, and a step of 0, stay in range.
    expect(await extractVideoFrames(file, { start: 8, end: 40, step: 0 }, () => {})).toHaveLength(2);
  });

  it("count a video's frames by rate", () => {
    expect(extractCount({ start: 0, end: 10, fps: 1 })).toBe(11);
    expect(extractCount({ start: 1, end: 1, fps: 2 })).toBe(1);
  });
});

describe("long recordings", () => {
  // Every case above, end to end, in one recording.
  const exif = seg(0xe1, [0x45, 0x78, 0x69, 0x66, 0, 0, ...jpeg([9, 9])]);
  const recording = bytes(
    [0x00, 0x13, 0xff, 0x37],
    jpeg([1, 2, 3]),
    [...jpeg([0xff, 0x00, 7, 0xff, 0xd3, 8], [exif]).slice(0, -2), 0xff, 0xff, 0xd9],
    [0xff, 0xd8, 0x00],
    jpeg([4]),
    jpeg([5, 6]),
    jpeg([7, 8, 9]).slice(0, -4),
  );

  it("are scanned a chunk at a time with the same result as all at once", async () => {
    const whole = await ranges(recording);
    expect(whole).toHaveLength(4);
    // Each chunk size splits a frame, a marker segment or an SOI across two reads.
    for (let chunk = 1; chunk <= recording.length + 1; chunk++) {
      expect(await mjpegFrameRanges(new Blob([recording]), chunk), `chunk ${chunk}`).toEqual(whole);
    }
  });

  it("are not held to the video size limit", () => {
    expect(maxImportBytes(new File([], "rec.mjpeg"))).toBe(MAX_RECORDING_BYTES);
    expect(maxImportBytes(new File([], "clip.mp4", { type: "video/mp4" }))).toBe(MAX_VIDEO_BYTES);
    // Ten minutes of the board at 45 fps, 1.5 MB a second at most.
    expect(MAX_RECORDING_BYTES).toBeGreaterThan(10 * 60 * 1.5 * 1024 * 1024);
  });
});
