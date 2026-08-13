# Workspace format

Everything PixelKit stores lives in one user-chosen folder (default
`~/PixelKit`, changeable in Settings; `PIXELKIT_WORKSPACE` overrides).
It is plain files — safe to back up, sync, or read from your own tools.
This document is the public contract; the implementation is
`engine/gd/store.py` and `engine/gd/workspace.py`.

## Layout

```
<workspace>/
  workspace.json                  # {"app": "pixelkit", "schemaVersion": 1}
  weights/                        # HF_HOME — model downloads (regenerable)
  projects/
    <project-slug>/               # a Project (container of datasets)
      project.json                # container metadata (name, dataset ids…)
      <dataset-slug>/             # one folder per dataset
        dataset.json              # metadata + labels + per-image index
        images/                   # ORIGINAL images only, never rewritten
        annotations/              # one JSON per image (see below)
        references/               # reference crops ("specific" datasets)
        augmentations/            # generated variants + transformed anns
        thumbs/                   # thumbnails/previews (regenerable cache)
        exports/                  # export zips land here
    <dataset-slug>/               # datasets outside any Project sit at root
      dataset.json
      ...
```

## Identity and naming

Folder names are **cosmetic slugs**; identity is the `"id"` field inside
`project.json` / `dataset.json`. Renaming a folder (or the slug drifting
from the display name) breaks nothing — the engine indexes by scanning
for the JSON files at startup. Derived (crop) datasets link to their
parent by id, so moves/renames survive.

## dataset.json vs annotations/

The SaaS build kept one monolithic `manifest.json` holding every box of
every image. The portable schema splits it:

- **`dataset.json`** — everything *except* per-image geometry: dataset
  name/id, labels + colours, thresholds, augmentation config, dataset
  type, references index, derived-dataset link, and a per-image index
  (filename, dimensions, blurhash, review flags…).
- **`annotations/<import_id>.json`** — the geometry for one image:
  `{detections, editedBoxes, timings}` (boxes, polygon masks, labels,
  scores, review verdicts).

Consequences you can rely on:

- A corrupt write costs one image's annotations, not the dataset.
- All writes are atomic (tmp file + rename); annotation files are only
  rewritten when their content changes.
- Diff/sync tools see per-image churn, not a giant file rewritten on
  every save.

## Caches

`thumbs/` and `weights/` are regenerable — deleting them costs a
re-download / re-render, never data. `images/` and `annotations/` are
the ground truth; treat them as the thing you back up.

## App config (not in the workspace)

Machine-local settings — workspace path, Hugging Face token, device
choice — live in the OS config dir (`%APPDATA%\PixelKit`,
`~/Library/Application Support/PixelKit`, `~/.config/pixelkit`), never
in the workspace, so a synced/shared workspace never leaks your token.
