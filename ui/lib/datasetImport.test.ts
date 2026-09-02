// @vitest-environment jsdom
// jsdom for DOMParser (VOC) + File; the COCO/YOLO parsers are pure.
import { describe, expect, it } from "vitest";

import {
  type CollectedFile,
  detectFormat,
  parseDataset,
  parseYamlNames,
  parseYoloLabelText,
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

describe("VOC", () => {
  it("still parses (regression guard for the dispatch rewrite)", async () => {
    const xml = `<annotation><filename>a.jpg</filename><size><width>10</width><height>20</height></size>
      <object><name>Cat</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>6</ymax></bndbox></object></annotation>`;
    const ds = await parseDataset([jpg("JPEGImages/a.jpg"), f("Annotations/a.xml", xml)]);
    expect(ds.format).toBe("voc");
    expect(ds.items[0].boxes[0]).toMatchObject({ label: "cat", x0: 1, y0: 2, x1: 5, y1: 6 });
  });
});
