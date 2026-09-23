// @vitest-environment jsdom
// jsdom for DOMParser (VOC) + File; the COCO/YOLO parsers are pure.
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  type CollectedFile,
  detectFormat,
  jpegOrientation,
  parseDataset,
  parseYamlNames,
  parseYoloLabelText,
  prepareItemForUpload,
} from "./datasetImport";

const f = (path: string, text = "", type?: string): CollectedFile => ({
  path,
  file: new File([text], path.split("/").pop()!, type ? { type } : undefined),
});
const jpg = (path: string) => f(path, "\xff\xd8\xff", "image/jpeg");

describe("detectFormat", () => {
  it("prefers VOC when any XML is present", () => {
    expect(detectFormat([jpg("a.jpg"), f("Annotations/a.xml")])).toBe("voc");
  });
  it("guesses COCO from JSON and YOLO from txt", () => {
    expect(detectFormat([jpg("a.jpg"), f("instances.json")])).toBe("coco");
    expect(detectFormat([jpg("a.jpg"), f("labels/a.txt"), f("classes.txt")])).toBe("yolo");
    expect(detectFormat([jpg("a.jpg")])).toBe("unknown");
  });
});

describe("COCO", () => {
  const coco = JSON.stringify({
    images: [
      { id: 1, file_name: "images/a.jpg", width: 100, height: 50 },
      { id: 2, file_name: "b.jpg", width: 40, height: 40 },
      { id: 3, file_name: "missing.jpg", width: 10, height: 10 },
    ],
    categories: [
      { id: 7, name: "Cat" },
      { id: 9, name: "dog" },
      { id: 11, name: "unused" },
    ],
    annotations: [
      { id: 100, image_id: 1, category_id: 7, bbox: [10, 5, 20, 10] },
      {
        id: 101,
        image_id: 1,
        category_id: 9,
        bbox: [0, 0, 30, 30],
        segmentation: [[0, 0, 30, 0, 30, 30]],
      },
      { id: 102, image_id: 1, category_id: 9, bbox: [1, 1, 0.5, 0.5] }, // degenerate
      { id: 103, image_id: 2, category_id: 7, bbox: [1, 1, 5, 5], segmentation: { counts: "rle", size: [40, 40] } },
    ],
  });

  it("parses boxes, masks, categories and pairs images", async () => {
    const files = [jpg("images/a.jpg"), jpg("b.jpg"), jpg("stray.png"), f("annotations/instances.json", coco)];
    const ds = await parseDataset(files);
    expect(ds.format).toBe("coco");
    expect(ds.classes).toEqual(["cat", "dog", "unused"]);
    const a = ds.items.find((i) => i.originalName === "a.jpg")!;
    expect(a.width).toBe(100);
    expect(a.boxes).toHaveLength(2);
    expect(a.boxes[0]).toMatchObject({ label: "cat", x0: 10, y0: 5, x1: 30, y1: 15, mask: null });
    expect(a.boxes[1].mask).toEqual({ polygons: [[[0, 0], [30, 0], [30, 30]]] });
    const b = ds.items.find((i) => i.originalName === "b.jpg")!;
    expect(b.boxes[0].mask).toBeNull(); // RLE dropped, box kept
    const stray = ds.items.find((i) => i.originalName === "stray.png")!;
    expect(stray.boxes).toEqual([]);
    expect(ds.stats).toMatchObject({ images: 3, annotated: 2, background: 1, boxes: 3, droppedBoxes: 1, unpairedAnnotations: 1 });
    expect(ds.warnings.join(" ")).toMatch(/RLE/);
  });

  it("ignores a non-COCO json and falls through to YOLO", async () => {
    const files = [jpg("a.jpg"), f("meta.json", JSON.stringify({ hello: 1 })), f("a.txt", "0 0.5 0.5 0.2 0.2"), f("classes.txt", "thing")];
    const ds = await parseDataset(files);
    expect(ds.format).toBe("yolo");
  });
});

describe("YOLO", () => {
  it("parses boxes in unit space with classes.txt", async () => {
    const files = [
      jpg("images/train/a.jpg"),
      jpg("images/val/a.jpg"), // same stem, different split: mirrored path must pair
      jpg("images/train/b.jpg"),
      f("labels/train/a.txt", "0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.3\n"),
      f("labels/val/a.txt", "1 0.5 0.5 0.5 0.5"),
      f("classes.txt", "cat\ndog\n"),
      f("README.txt", "This is a dataset.\nNot labels."),
    ];
    const ds = await parseDataset(files);
    expect(ds.format).toBe("yolo");
    expect(ds.classes).toEqual(["cat", "dog"]);
    const trainA = ds.items.find((i) => i.file === files[0].file)!;
    expect(trainA.width).toBe(1);
    expect(trainA.height).toBe(1);
    expect(trainA.boxes[0]).toMatchObject({ label: "cat" });
    expect(trainA.boxes[0].x0).toBeCloseTo(0.4);
    expect(trainA.boxes[0].y0).toBeCloseTo(0.3);
    expect(trainA.boxes[0].x1).toBeCloseTo(0.6);
    expect(trainA.boxes[0].y1).toBeCloseTo(0.7);
    expect(trainA.boxes[1]).toMatchObject({ label: "dog", x0: 0.1, y0: 0.1, x1: 0.3, y1: 0.3 });
    expect(trainA.boxes[1].mask).toEqual({ polygons: [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]] });
    const valA = ds.items.find((i) => i.file === files[1].file)!;
    expect(valA.boxes[0].label).toBe("dog");
    const b = ds.items.find((i) => i.file === files[2].file)!;
    expect(b.boxes).toEqual([]);
    expect(ds.stats).toMatchObject({ images: 3, annotated: 2, background: 1, boxes: 3 });
    expect(ds.warnings.join(" ")).toMatch(/README|not label rows/);
  });

  it("falls back to class_N names and rejects folders with no label rows", async () => {
    const ds = await parseDataset([jpg("a.jpg"), f("a.txt", "3 0.5 0.5 0.1 0.1")]);
    expect(ds.classes).toEqual(["class_3"]);
    await expect(parseDataset([jpg("a.jpg"), f("notes.txt", "hello")])).rejects.toThrow(/No YOLO label/);
  });

  it("parseYoloLabelText distinguishes prose from labels", () => {
    expect(parseYoloLabelText("hello world")).toBeNull();
    expect(parseYoloLabelText("")).toEqual({ rows: [], bad: 0 });
    const r = parseYoloLabelText("0 0.5 0.5 1 1\nbroken line\n-1 0.5 0.5 0.1 0.1")!;
    expect(r.rows).toHaveLength(1);
    expect(r.bad).toBe(2);
  });

  it("parseYamlNames handles inline, list and mapping forms", () => {
    expect(parseYamlNames("nc: 2\nnames: ['a', 'b']\n")).toEqual(["a", "b"]);
    expect(parseYamlNames("names:\n  - Cat\n  - dog\ntrain: x")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("names:\n  1: dog\n  0: cat\n")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("nc: 2")).toEqual([]);
  });
});

describe("sweep regressions", () => {
  it("COCO: folder-prefixed paths pair by suffix and twins never cross splits", async () => {
    // The folder picker prefixes every path with the chosen folder name.
    const train = JSON.stringify({
      images: [{ id: 1, file_name: "train/a.jpg", width: 10, height: 10 }],
      categories: [{ id: 1, name: "cat" }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [1, 1, 5, 5] }],
    });
    const val = JSON.stringify({
      images: [{ id: 1, file_name: "val/a.jpg", width: 10, height: 10 }],
      categories: [{ id: 2, name: "dog" }],
      annotations: [{ id: 9, image_id: 1, category_id: 2, bbox: [2, 2, 3, 3] }],
    });
    const files = [jpg("ds/train/a.jpg"), jpg("ds/val/a.jpg"), f("ds/instances_train.json", train), f("ds/instances_val.json", val)];
    const ds = await parseDataset(files);
    const t = ds.items.find((i) => i.file === files[0].file)!;
    const v = ds.items.find((i) => i.file === files[1].file)!;
    expect(t.boxes.map((b) => b.label)).toEqual(["cat"]);
    expect(v.boxes.map((b) => b.label)).toEqual(["dog"]);
    expect(ds.stats.unpairedAnnotations).toBe(0);
  });

  it("COCO: a bare file name that matches two files is ambiguous, not last-wins", async () => {
    const j = JSON.stringify({
      images: [{ id: 1, file_name: "a.jpg", width: 10, height: 10 }],
      categories: [{ id: 1, name: "cat" }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [1, 1, 5, 5] }],
    });
    const ds = await parseDataset([jpg("ds/train/a.jpg"), jpg("ds/val/a.jpg"), f("ds/x.json", j)]);
    expect(ds.stats.unpairedAnnotations).toBe(1);
    expect(ds.items.every((i) => i.boxes.length === 0)).toBe(true);
  });

  it("YAML names: column-0 list, trailing comment, flow mapping, quoted keys, multi-line list", () => {
    expect(parseYamlNames("nc: 2\nnames:\n- cat\n- dog\nval: x")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("names: ['cat', 'dog']  # classes")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("names: {0: cat, 1: dog}")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("names:\n  '1': dog\n  \"0\": cat\n")).toEqual(["cat", "dog"]);
    expect(parseYamlNames("names: [cat,\n  dog]\nnc: 2")).toEqual(["cat", "dog"]);
  });

  it("YOLO: confidence rows are boxes with a score; malformed rows are not prose", () => {
    const r = parseYoloLabelText("0 0.5 0.5 0.2 0.2 0.93\n")!;
    expect(r.rows).toHaveLength(1);
    expect(r.rows[0].score).toBeCloseTo(0.93);
    const t = parseYoloLabelText("0 0.5 0.5 0.2\n")!; // truncated row: still a label file
    expect(t).toEqual({ rows: [], bad: 1 });
    expect(parseYoloLabelText("Licensed under CC-BY.\nSee README.")).toBeNull();
  });
});

describe("VOC", () => {
  it("still parses (regression guard for the dispatch rewrite)", async () => {
    const xml = `<annotation><filename>a.jpg</filename><size><width>10</width><height>20</height></size>
      <object><name>Cat</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>6</ymax></bndbox></object></annotation>`;
    const ds = await parseDataset([jpg("JPEGImages/a.jpg"), f("Annotations/a.xml", xml)]);
    expect(ds.format).toBe("voc");
    expect(ds.items[0].boxes[0]).toMatchObject({ label: "cat", x0: 1, y0: 2, x1: 5, y1: 6 });
  });
});

describe("pairing and orientation regressions", () => {
  const voc = (fn: string | null, label: string) =>
    `<annotation>${fn ? `<filename>${fn}</filename>` : ""}<size><width>100</width><height>100</height></size>` +
    `<object><name>${label}</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>`;
  const pairs = (files: CollectedFile[], ds: Awaited<ReturnType<typeof parseDataset>>) =>
    ds.items.map((it) => [files.find((cf) => cf.file === it.file)?.path, it.boxes[0]?.label ?? null]);

  it("VOC: split twins pair within their own split", async () => {
    const files = [
      jpg("ds/train/JPEGImages/0001.jpg"),
      jpg("ds/val/JPEGImages/0001.jpg"),
      f("ds/train/Annotations/0001.xml", voc("0001.jpg", "cat")),
      f("ds/val/Annotations/0001.xml", voc("0001.jpg", "dog")),
    ];
    expect(pairs(files, await parseDataset(files))).toEqual([
      ["ds/train/JPEGImages/0001.jpg", "cat"],
      ["ds/val/JPEGImages/0001.jpg", "dog"],
    ]);
  });

  it("VOC: twins beside their XMLs, with no <filename>, pair too", async () => {
    const files = [jpg("ds/train/0001.jpg"), f("ds/train/0001.xml", voc(null, "cat")),
                   jpg("ds/val/0001.jpg"), f("ds/val/0001.xml", voc(null, "dog"))];
    expect(pairs(files, await parseDataset(files))).toEqual([
      ["ds/train/0001.jpg", "cat"],
      ["ds/val/0001.jpg", "dog"],
    ]);
  });

  it("VOC: a.jpg and a.png are both imported", async () => {
    const files = [jpg("JPEGImages/a.jpg"), f("JPEGImages/a.png", "PNG", "image/png"),
                   f("Annotations/a.xml", voc("a.jpg", "cat"))];
    expect(pairs(files, await parseDataset(files))).toEqual([
      ["JPEGImages/a.jpg", "cat"],
      ["JPEGImages/a.png", null],
    ]);
  });

  it("COCO: a captions file does not take the images' boxes away", async () => {
    const img = { id: 139, file_name: "000000000139.jpg", width: 640, height: 426 };
    const captions = { images: [img], annotations: [{ id: 1, image_id: 139, caption: "A room." }] };
    const instances = {
      images: [img],
      annotations: [{ id: 10, image_id: 139, category_id: 1, bbox: [10, 20, 100, 200], iscrowd: 0 }],
      categories: [{ id: 1, name: "person" }],
    };
    const files = [
      jpg("coco/val2017/000000000139.jpg"),
      f("coco/annotations/captions_val2017.json", JSON.stringify(captions)),
      f("coco/annotations/instances_val2017.json", JSON.stringify(instances)),
    ];
    const ds = await parseDataset(files);
    expect(ds.stats.boxes).toBe(1);
    expect(ds.items[0].boxes[0].label).toBe("person");
  });

  describe("a photo annotated on its stored, unrotated frame", () => {
    // A JPEG header whose EXIF says: shown turned a quarter clockwise (6).
    const exifJpeg = (orientation: number) => {
      const tiff = [0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, orientation, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00];
      const app1 = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00, ...tiff];
      const len = app1.length + 2;
      return new Uint8Array([0xff, 0xd8, 0xff, 0xe1, len >> 8, len & 0xff, ...app1, 0xff, 0xda, 0x00, 0x02, 0xff, 0xd9]);
    };
    const g = globalThis as unknown as { createImageBitmap?: unknown };
    let saved: unknown;
    beforeAll(() => {
      saved = g.createImageBitmap;
      g.createImageBitmap = async () => ({ width: 3000, height: 4000, close() {} }); // decoded upright
    });
    afterAll(() => {
      g.createImageBitmap = saved;
    });

    it("reads the EXIF orientation from the header", async () => {
      expect(await jpegOrientation(new Blob([exifJpeg(6)]))).toBe(6);
      expect(await jpegOrientation(new Blob([exifJpeg(8)]))).toBe(8);
      expect(await jpegOrientation(new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xda])]))).toBe(1);
    });

    it("turns its boxes with the picture instead of stretching them", async () => {
      const file = new File([exifJpeg(6)], "IMG_0001.jpg", { type: "image/jpeg" });
      const out = await prepareItemForUpload({
        file,
        originalName: "IMG_0001.jpg",
        boxes: [{ id: "b", label: "cat", x0: 100, y0: 200, x1: 500, y1: 600, score: null }],
        width: 4000, // the stored frame, as labelImg reports it
        height: 3000,
      }, null);
      expect(out.boxes[0]).toMatchObject({ x0: 2400, y0: 100, x1: 2800, y1: 500 });
    });
  });
});
