import { describe, expect, it } from "vitest";

import { splitLabels } from "./labelInput";

describe("splitLabels", () => {
  it("splits a pasted comma-separated list", () => {
    expect(splitLabels("pothole, crack, manhole")).toEqual(["pothole", "crack", "manhole"]);
  });

  it("splits on newlines and drops empty parts and trailing full stops", () => {
    expect(splitLabels("bolt.\n, washer,,\n")).toEqual(["bolt", "washer"]);
  });

  it("keeps a single label with inner spaces and dots", () => {
    expect(splitLabels("  st. bernard ")).toEqual(["st. bernard"]);
  });

  it("returns nothing for separators only", () => {
    expect(splitLabels(" , \n ")).toEqual([]);
  });
});
