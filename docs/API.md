# Local API: training and trained models

The engine listens on `http://127.0.0.1:8001` (the packaged app starts it;
`pixelkit serve` does the same from a checkout). Everything below is what
the Train card uses, so a script, NeuroStudio, or a CI job can drive the
same path: dataset in, deployable model out. No authentication in the
portable build.

`{pid}` is a dataset id (the folder name under `<workspace>/projects/`,
also in the app's URL). `{run}` is a run id, `YYYYMMDDTHHMMSSZ`.

## Pre-flight

```
GET /api/v2/projects/{pid}/train/preflight?imgsz=256&split=0.8&size=nano&augmentations=1&evalProject=
```

What a run with these options would use and what would go wrong,
without starting one:

```json
{
  "ok": true,
  "blockers": [],
  "warnings": ["'bolt' has only 7 training box(es): expect poor recall"],
  "images": {"train": 312, "val": 78, "augmented": 1248, "unlabelled": 4},
  "classes": [{"name": "bolt", "trainImages": 6, "valImages": 2, "trainBoxes": 7,
               "valBoxes": 2, "augmentedBoxes": 28, "smallBoxesPct": 0}],
  "imgsz": 256, "size": "nano", "evalProject": null
}
```

`blockers` stop the run (fewer than two labelled images, a held-out
dataset with none of these labels). `warnings` are recorded in the run's
report: classes with few boxes, classes the val split cannot measure,
objects under 8 px at the input size, byte-identical images on both
sides of the split.

## Start a run

```
POST /api/jobs
{"kind": "train_yolox", "project": "{pid}", "params": {...}}
```

| param | default | meaning |
|---|---|---|
| `epochs` | 50 | float epochs (1-500); 8 quantisation-aware epochs follow |
| `imgsz` | 256 | square input: 128, 192, 256, 320 or 416 |
| `size` | `nano` | `nano` (0.9 M parameters) or `tiny` (5 M) |
| `split` | 0.8 | train share, 0.5-0.95; the rest is held out per class |
| `augmentations` | true | train on the dataset's generated augmentations |
| `initRun` | | warm-start from a previous run of this dataset |
| `evalProject` | | a second dataset id used as the held-out set |
| `resumeRun` | | continue an interrupted run; other params are ignored |
| `conf` | | fix the confidence threshold instead of calibrating it |
| `device` | engine device | e.g. `cuda:1` to train on a second GPU |

Returns `{"jobId": ...}`. One training job per dataset at a time (409
otherwise). Progress is on `GET /api/jobs/{jobId}`
(`progress.index/total` = epoch, `progress.image` = loss, AP@0.5 and the
time left) and on the status bar. `DELETE
/api/v2/projects/{pid}/jobs/{jobId}` cancels; a cancelled run keeps its
checkpoint and shows as interrupted.

## Runs

```
GET /api/v2/projects/{pid}/models
```

Newest first. Each entry: `run`, `model` (`yolox_nano` / `yolox_tiny`),
`size`, `status` (`done`, `training`, `interrupted`, `failed`,
`rejected`), `resumable`, `epochsDone`, `totalEpochs`, `images`,
`ap50Fp32`, `ap50Int8`, `threshold` (`conf`, `f1`, `precision`,
`recall`, `calibrated`), `neuron6` (the ST Edge AI build summary),
`hasExport`, `failuresCount`, `initRun`, `resumed`, and, for a run in
progress, `history`.

```
GET /api/v2/projects/{pid}/models/{run}
```

The same entry plus `report` (the run's `report.json`: `classes`,
`history`, `threshold` with the full sweep, `classReport`, `confusion`,
`confusionLabels`, `failures`, `per_class_fp32`, `per_class_int8`,
`warnings`, `notes`, `neuron6`, the input contract) or, while it trains,
`live` (`history.json`).

```
GET /api/v2/projects/{pid}/models/{run}/export            -> export.zip
GET /api/v2/projects/{pid}/models/{run}/failures/{file}   -> annotated JPEG
POST /api/v2/projects/{pid}/models/{run}/resume  {"device": "cuda:1"}   -> a train job
DELETE /api/v2/projects/{pid}/models/{run}
```

`export.zip` holds the int8 ONNX, the labels header (with the calibrated
`_CONF_THRESHOLD`), the sketch snippet and `report.json`; see
[TRAINING.md](TRAINING.md) for what is in it and how the core consumes
it.

## Label images with a trained model

```
POST /api/jobs
{"kind": "label_model", "project": "{pid}",
 "params": {"run": "{run}", "conf": 0.35, "overwrite": false}}
```

Runs the run's exported int8 model over the dataset's unlabelled images
(all auto-labelled images too with `overwrite`; images edited by hand are
never touched) at the run's calibrated threshold unless `conf` is given.
Detections land in the manifest like the SAM 3 labeller's, so the
editor, exports and the next training run read them the same way.

## Headless

```bash
pixelkit train <dataset id or name> --epochs 50 --input 256 --size nano --split 0.8 \
    [--no-augmentations] [--eval-dataset NAME] [--from RUN] [--resume RUN] \
    [--conf 0.4] [--device cuda:1] [--out ./exports]
```

Prints the pre-flight, per-epoch progress, and the run summary; copies
`export.zip` into `--out`.

## Files on disk

`<dataset>/models/<yolox_nano|yolox_tiny>/<run>/`: `report.json`,
`history.json` (per-epoch, written as it trains), `ckpt_last.pt` (only
while a run is unfinished; its presence is what makes a run resumable),
`best.pt`, `<slug>_fp32.onnx`, `<slug>_int8.onnx`, `<slug>_labels.h`,
`<slug>_snippet.ino.txt`, `export.zip`, `failures/` + `failures.json`,
`neuron6/` (compiler log and report).
