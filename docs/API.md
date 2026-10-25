# Local API: training and trained models

The engine listens on `http://127.0.0.1:8001`. The desktop app starts it;
`pixelkit serve` starts it from a checkout. These are the endpoints the Train
card uses. There is no authentication. The engine answers only requests
addressed to 127.0.0.1 or localhost, and refuses a POST, PUT, PATCH or DELETE
whose `Origin` is another site; scripts send no `Origin`.

`{pid}` is a dataset id: the `id` field of the dataset's `dataset.json`, also
in the app's URL, `/app/<id>`. `{run}` is a run id, `YYYYMMDDTHHMMSSZ`.

## Pre-flight

```
GET /api/v2/projects/{pid}/train/preflight?imgsz=256&split=0.8&size=nano&augmentations=1&evalProject=
```

Reports what a run with these options would use and what is wrong, without
starting one:

```json
{
  "ok": true,
  "blockers": [],
  "warnings": ["'bolt' has only 7 training box(es): expect poor recall"],
  "images": {"train": 312, "val": 78, "augmented": 1248, "augmentedStale": 0,
             "unlabelled": 4, "background": 40},
  "classes": [{"name": "bolt", "trainImages": 6, "valImages": 2, "trainBoxes": 7,
               "valBoxes": 2, "augmentedBoxes": 28, "smallBoxesPct": 0}],
  "imgsz": 256, "size": "nano", "evalProject": null
}
```

`background` is the number of background images in train and val: images
without boxes, trained on as negatives. `blockers` stop the run: fewer than
two labelled images, no image with boxes, a held-out dataset with none of
these labels. `warnings` are recorded in the run's report: classes with few
boxes, classes the validation split cannot measure, objects under 8 px at
the input size, byte-identical images on both sides of the split.

## Starting a run

```
POST /api/jobs
{"kind": "train_yolox", "project": "{pid}", "params": {...}}
```

| Param | Default | Meaning |
|---|---|---|
| `epochs` | 50 | float epochs, 1 to 500; 8 quantisation-aware epochs follow |
| `imgsz` | 256 | square input: 128, 192, 256, 320 or 416 |
| `size` | `nano` | `nano` (0.9 M parameters) or `tiny` (5 M) |
| `split` | 0.8 | train share, 0.5 to 0.95; the rest is held out per class |
| `augmentations` | true | train on the dataset's generated augmentations |
| `initRun` | | start from a previous run of this dataset |
| `evalProject` | | a second dataset id used as the held-out set |
| `resumeRun` | | continue an interrupted run; other params are ignored |
| `conf` | | a fixed confidence threshold instead of a calibrated one |
| `device` | | for example `cuda:1`; by default a second GPU when there is one, else the engine's device |

Returns `{"jobId": ...}`. A dataset trains one run at a time; a second
request returns 409, also when another process is training it or while
augmentations are being generated.

`GET /api/jobs/{jobId}` reports progress: `progress.index/total` is the
epoch, and `progress.image` the loss, AP@0.5 and time remaining. Once the
job is `done`, it also carries `result`: `run`, `model`, `size`,
`ap50_fp32`, `ap50_int8`, `threshold`.

`DELETE /api/v2/projects/{pid}/jobs/{jobId}` cancels. A cancelled run keeps
its last checkpoint and shows as interrupted, or as failed if no epoch had
finished.

## Runs

```
GET /api/v2/projects/{pid}/models
```

Newest first. Each entry: `run`, `model` (`yolox_nano` or `yolox_tiny`),
`size`, `status` (`done`, `training`, `interrupted`, `failed`, `rejected`),
`resumable`, `epochsDone`, `totalEpochs`, `images`, `ap50Fp32`, `ap50Int8`,
`threshold`, `neuron6` (the ST Edge AI build summary), `hasExport`,
`failuresCount`, `hasPredictions`, `initRun`, `resumed`, and `history` for a
run in progress.

`threshold` has `conf`, `f1`, `precision`, `recall`, `calibrated` and
`fallback`, which is true when the model found nothing on val and the
default was written.

```
GET /api/v2/projects/{pid}/models/{run}
```

The same entry plus `report`, the run's `report.json` (`classes`,
`history`, `threshold` with the full sweep, `classReport`, `confusion`,
`confusionLabels`, `failures`, `per_class_fp32`, `per_class_int8`,
`warnings`, `notes`, `neuron6`, the input contract), or `live`, the run's
`history.json`, while it trains.

```
GET    /api/v2/projects/{pid}/models/{run}/export            -> export.zip
GET    /api/v2/projects/{pid}/models/{run}/failures/{file}   -> annotated JPEG
GET    /api/v2/projects/{pid}/models/{run}/predictions       -> predictions.json
POST   /api/v2/projects/{pid}/models/{run}/resume  {"device": "cuda:1"}   -> a train job
DELETE /api/v2/projects/{pid}/models/{run}
```

`predictions.json` holds every validation image with its ground truth, in
image pixels:

```
{"conf", "classes", "images": [{"project", "image", "width", "height",
  "truth": [{"box", "class", "missed"}],
  "predicted": [{"box", "class", "score", "kind"}]}]}
```

`kind` is `tp`, `wrong` (right place, wrong class) or `fp`, at the shipped
threshold `conf`. `project` is the held-out dataset for a run scored on one.

`export.zip` is a `<slug>/` sketch folder with the sketch, the int8 ONNX
model, the labels header with the calibrated `_CONF_THRESHOLD`, `report.json`,
and `NOTICE.txt` and `LICENSE.txt` for the YOLOX network the model is built
on. [TRAINING.md](TRAINING.md) describes its contents and how
the core builds it.

## Labelling with a trained model

```
POST /api/jobs
{"kind": "label_model", "project": "{pid}",
 "params": {"run": "{run}", "conf": 0.35, "overwrite": false}}
```

Runs the run's int8 model over the dataset's unlabelled images, at the
run's calibrated threshold unless `conf` is given. With `overwrite`, images
labelled automatically are labelled again too. Images edited by hand,
including ones emptied by hand, are never changed. Detections are stored
like the SAM 3 labeller's, so the editor, exports and training read them the
same way. The job's `result` is `{"labelled": n, "run", "conf", "cancelled"}`.

## Background images

```
PUT /api/v2/projects/{pid}/imports/{importId}  {"background": true}
```

Marks an image as background: reviewed, with nothing in it to label. Its
boxes are dropped, it counts as labelled, both kinds of labelling job skip
it, and training uses it as a negative. `{"background": false}` unmarks it,
and so does saving boxes onto it (`{"editedBoxes": [...]}`). The dataset's
`/overview` carries the flag per image. Importing a labelled dataset marks
the images whose annotation lists no objects.

## Command line

```bash
pixelkit train <dataset id or name> --epochs 50 --input 256 --size nano --split 0.8 \
    [--no-augmentations] [--eval-dataset NAME] [--from RUN] [--resume RUN] \
    [--conf 0.4] [--device cuda:1] [--out ./exports]
```

Prints the pre-flight, per-epoch progress and the run summary, and copies
`export.zip` into `--out`.
