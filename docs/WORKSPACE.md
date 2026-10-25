# Workspace format

PixelKit stores everything in one workspace folder: `~/PixelKit` by default,
set in Settings or by `PIXELKIT_WORKSPACE`. Its contents are plain files.
This document is the public contract. The implementation is
`engine/gd/store.py` and `engine/gd/workspace.py`.

## Layout

```
<workspace>/
  workspace.json              {"app": "pixelkit", "schemaVersion": 1}
  events.db                   activity and job history (SQLite)
  weights/                    model downloads, regenerable
    yolox/                    YOLOX COCO checkpoints
  _storage/containers/<project id>/cover.jpg
                              a project's cover
  projects/
    <project-slug>/           a project: a container of datasets
      project.json            id, name, dataset ids
      <dataset-slug>/
        dataset.json          id, name, labels, settings, per-image index
        images/               imported images, never rewritten
        annotations/          one JSON per image
        references/           reference images
        augmentations/        generated copies and their annotations
        augment_backgrounds/  uploaded backgrounds for augmentation
        snapshots/            annotation snapshots
        models/
          yolox_nano/<run>/   trained model and Neuro N6 export
          yolox_tiny/<run>/
        cover_upload.jpg      an uploaded cover
        thumbs/               labelled previews, regenerable
        image_embeddings/     DINOv2 embeddings, regenerable
        *_first*.json, workspace_card_v1.json, dataset_stats*.json
                              payload caches, regenerable
    <dataset-slug>/           a dataset outside any project
```

A run folder's files are listed in [TRAINING.md](TRAINING.md#run-folder).

## Identity

Identity is the `"id"` field in `project.json` and `dataset.json`. Folder
names are slugs with no meaning: the engine finds datasets by scanning for
these files at startup, so renaming a folder breaks nothing. A derived crop
dataset refers to its parent by id.

## dataset.json and annotations/

- `dataset.json` holds everything except per-image geometry: name and id,
  labels and colours, thresholds, augmentation settings, the references
  index, the parent link of a derived dataset, and an index of images with
  filename, dimensions, blurhash and review flags.
- `images/` holds each imported image under a generated name. A JPEG file is
  stored as uploaded; other formats are stored as JPEG at quality 90.
  Openverse images keep the format they were downloaded in.
- `annotations/<import_id>.json` holds one image's geometry:
  `{detections, editedBoxes, timings}`: boxes, polygon masks, labels,
  scores and review verdicts.

Every write is atomic, a temporary file followed by a rename. An annotation
file is rewritten only when its content changes, and a corrupt file affects
one image.

## Snapshots

`snapshots/<UTC timestamp>-<hash>.zip` holds a copy of `dataset.json` and
`annotations/`, without images. Snapshots are created from the dataset's
settings panel. Restoring one first takes a snapshot of the current state.

## Caches

`weights/` and, in each dataset, `thumbs/`, `image_embeddings/` and the files
`overview_first100.json`, `initial_first20.json`, `workspace_card_v1.json`,
`dataset_stats.json` and `dataset_stats_lite.json` can be deleted; they are
downloaded or rebuilt. Update on the Augmentations tab makes `augmentations/`
again, with new random values. The rest is the data to back up:
`events.db`, `_storage/`, `project.json`, and in each dataset
`dataset.json`, `images/`, `annotations/`, `references/`,
`augment_backgrounds/`, `cover_upload.jpg`, `snapshots/` and `models/`.

Model downloads go to `weights/` unless `HF_HOME` is set in the environment.

## App config

Machine-local settings, which include the workspace path, the Hugging Face
token and the device, are stored outside the workspace:

| OS | Folder |
|---|---|
| Windows | `%APPDATA%\PixelKit` |
| macOS | `~/Library/Application Support/PixelKit` |
| Linux | `$XDG_CONFIG_HOME/pixelkit` or `~/.config/pixelkit` |

`PIXELKIT_CONFIG_DIR` overrides the folder.
