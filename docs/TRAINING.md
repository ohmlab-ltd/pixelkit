# Training and Neuro N6 export

PixelKit trains a YOLOX detector (Nano or Tiny) on a dataset and exports
it for the OhmLab Neuro N6 (STM32N657, Neural-ART NPU). Everything runs
locally. The local API behind the Train card is in [API.md](API.md).

## What you get

`<dataset>/models/<yolox_nano|yolox_tiny>/<run>/`:

| File | Purpose |
|---|---|
| `<slug>_int8.onnx` | the deployable model: full-integer QDQ int8, raw head |
| `<slug>_labels.h` | `<SLUG>_NUM_CLASSES`, `<SLUG>_INPUT_W/H`, the calibrated `<SLUG>_CONF_THRESHOLD`, `<SLUG>_NMS_THRESHOLD`, `<SLUG>_CLASSES[]` |
| `<slug>.ino` | the sketch: declares the camera, feeds the model at its input size and binds the core's `PostProcess_PixelKitYOLOX`, which publishes the boxes to Neuro Studio; the confidence threshold is a live slider there |
| `report.json` | AP@0.5 float vs int8 (overall and per class), threshold sweep, per-class precision/recall, confusion matrix, per-epoch history, input contract, NPU build facts |
| `export.zip` | a `<slug>/` sketch folder with the sketch, the model, the labels header and `report.json`: unzip it and open the sketch in the Arduino IDE |
| `failures/`, `failures.json` | the worst val images with misses, spurious boxes and wrong classes drawn |
| `predictions.json` | every val image's ground truth beside the int8 model's boxes at the shipped threshold, in the image's pixels |
| `history.json` | per-epoch loss and AP@0.5, written as the run trains |
| `neuron6/` | ST Edge AI compiler log and generate report |
| `<slug>_fp32.onnx`, `best.pt` | the float model and checkpoint, for re-export or inspection |
| `ckpt_last.pt` | only while a run is unfinished: what Resume continues from |

## What trains, and what is measured

The training set is every labelled image the split puts in train, plus
that image's generated augmentations (from the Augmentations tab), with
their transformed boxes. Augmentations of a val image are never used, so
the val score is never inflated by a copy of a val picture.

Images marked **background** train too, as negatives: pictures of the
scene with nothing in it, which teach the model where not to fire. A
fixed camera that reports objects in an empty room needs exactly these.
Mark an image with no boxes as background in the viewer, or many at once
from the gallery's Select mode; an imported dataset's images whose
annotation lists no objects arrive marked. A background image counts as
labelled, the labelling runs leave it alone, and drawing a box on it
unmarks it. Images with no boxes that are not marked stay unlabelled
and are not used.

The split is per class rather than per image: every class with two or
more images gets at least one val image and keeps at least one train
image, and each class is held out at the chosen ratio (60 to 90% of its
images train). Background images are split the same way, as a class of
their own, so the val score also counts boxes found where there is
nothing. It is keyed on filenames, so the same dataset splits the
same way every time. One exception: an image that also shows a class
with a single image stays in train, because learning that class beats
measuring another; a class left unmeasured that way is named by the
pre-flight check. Alternatively a second dataset can be the held-out set:
everything in this dataset trains, everything labelled in that one is
scored, labels matched by name. That is the honest number for a camera
the model has never seen.

Before a run starts, a pre-flight check says what will be used and
what will go wrong: too few labelled images, or background images with
no boxed image to learn from (blocks), classes with fewer
than ten boxes, classes the val split cannot measure, objects under 8
px at the chosen input size, byte-identical images on both sides of the
split. Warnings are recorded in the report.

## Nano or Tiny

| | parameters | int8 weights | MACs at 256x256 | when |
|---|---|---|---|---|
| YOLOX-Nano | 0.9 M | about 0.9 MB | 0.27 G | the default; a few classes, a fixed camera |
| YOLOX-Tiny | 5.1 M | about 5 MB | 1.3 G | Nano's accuracy is not enough; still under 20% of the weights pool |

Both use the same stem, head, quantisation and export path; the compile
report says what each costs on the board. A run can also start from a
previous run's weights instead of the COCO checkpoint (Start from), for
a second pass on a grown dataset.

## Batch size, interruptions

The batch is sized from free VRAM at the start (about 70 MB per image
for Nano and 150 MB for Tiny at 256x256, 60% of what is free, capped at
64) and then capped so an epoch has at least 15 optimiser steps (a
240-image dataset trains at batch 16, not 64: with three steps per
epoch a 50-epoch run barely moves). If the card still runs out of
memory the batch is halved and the epoch restarts; the report records
it. A checkpoint is written after every float epoch, so an interrupted
run (the app closed, a crash, Cancel training in the card) shows as
Interrupted and resumes from its last epoch with its own settings; an
interruption during the int8 fine-tuning restarts that short phase
from the float result, and a cancel before the first epoch completes
shows as failed. On a machine with two GPUs the run goes to the card
labelling is not using. The Train card shows the loss and AP@0.5 curves
as the run trains, with the time left.

## The input contract

The core's build tells ST Edge AI `--input-data-type uint8
--inputs-ch-position chlast`, because the camera DMAs packed RGB888 bytes
straight into the model's input buffer. That is only correct when the
graph input is quantised asymmetrically over [0,1]: scale 1/255,
zero-point -128. PixelKit trains on [0,1] tensors, the QAT graph pins
the input fake-quant to exactly those parameters, and the export checks
the file it wrote: a calibrated input range outside [0,1] (a symmetric
or a [0,255] model) is refused rather than rewritten, because a post-hoc
rewrite of a finished network measures worse than the bug it fixes.

## The head

The graph outputs one raw head tensor, `N x (5 + classes)` rows of
`[dx, dy, dw, dh, obj, cls...]`. The grid decode runs on the M55 in the
core's `yolox_pp.c`, because YOLOX's in-graph decode exports
`ScatterND`/`Expand`/`Where`, which ST Edge AI cannot shape-infer. The
core's `PostProcess_PixelKitYOLOX` wraps that decoder so the boxes reach
Neuro Studio and the display the way every bundled model's do. The
Focus stem is replaced by a 6x6 stride-2 conv (mathematically identical;
COCO stem weights are remapped) so no `Slice` runs in software.

## Why the export is quantisation-aware

Post-training quantisation of this network loses a lot: on VOC2007
(5011 images, 20 classes) the float model scores 0.550 AP@0.5 and its
int8 PTQ export 0.339, and every class is hurt. Bisecting the loss
showed the weights quantise for free (+0.1 with per-channel int8) and
the entire gap is per-tensor activation quantisation in the depthwise
backbone: outlier channels dominate each tensor's range. Better
calibration only recovers part of it (MinMax -17, moving average -11,
percentile 99.9 -8). So after the float run PixelKit fine-tunes for
`qat_epochs` (default 8) with int8 simulated at every tensor between
ops (weights per-channel, activations per-tensor with moving-average
ranges, BatchNorm folded, input pinned to the contract) and exports the
QDQ graph straight from that model. The exported graph carries int8
weights, int32 biases, and Q/DQ around every op, which is the shape
ST Edge AI maps entirely to hardware.

## NPU verification (report.json `neuron6`)

When ST Edge AI Core is installed (`C:\ST\STEdgeAI\<ver>`, or
`NEURON6_STEDGEAI`), every export is compiled exactly as the core's
`#pragma neuron6` pipeline compiles it, against the board's memory map:

```json
"neuron6": {
  "verified": true,
  "epochsTotal": 122, "epochsSoftware": 0, "epochsHybrid": 3, "epochsHardware": 119,
  "weightsBytes": 919744, "poolUsedPct": 3.44, "poolRemainingBytes": 25818944,
  "activationsBytes": 458752, "macc": 271599330
}
```

`epochsSoftware` is the number that matters: 0 means the whole network
runs on the NPU. `poolRemainingBytes` is headroom in the external-flash
weights pool the core's memory map declares (25.5 MB, ending where OTA
slot B begins). PixelKit compiles against a copy of that map, and a test
checks the copy against a core installed on the machine. A model the
compiler rejects is not
exported; the reason is recorded and shown in the app. Without the
compiler the export still succeeds and says `verified: false`.

## The threshold, and how the model fails

The decoder on the board needs one confidence threshold. After the int8
export is scored, PixelKit sweeps the threshold from 0.05 to 0.95 on
the val predictions and writes the one with the best F1 (all classes
pooled; the middle of a plateau when several tie) into the labels
header, with the precision and recall it gives. The report also carries
per-class precision and recall at that threshold, a confusion matrix
(truth by prediction, with a background row and column for spurious
boxes and misses) and a gallery of the val images with the most errors:
green is the truth, thin blue a correct detection, yellow a miss, red a
spurious box, orange the right place with the wrong class. That gallery
is where the next labelling hour should go. Beyond the worst few, Details
can show every val image twice, side by side: the ground truth it was
scored against, and what the int8 model reports at the shipped
threshold.

On the board the threshold is not fixed: the exported sketch registers
it with the core's live tuning (`n6_tune_float`), so Neuro Studio shows
a **confidence** slider while the sketch runs. Drag it on your own scene
to trade missed objects against false ones, then write the value into
`<SLUG>_CONF_THRESHOLD`; the slider itself is not saved on the board.

## Label with your own model

A finished run can pre-label the dataset's unlabelled images (Label
images on the run): the exported int8 model, run on the CPU at its
calibrated threshold, writes boxes the way the SAM 3 labeller does.
Hand-edited images are never touched. Review, correct, retrain.

## The recipe, validated on Pascal VOC 2007

Defaults: 50 float epochs plus 8 quantisation-aware epochs, 256x256,
batch from free VRAM, SGD with warmup and cosine decay, hflip + HSV +
scale jitter, COCO warm start. With `qat_epochs=0` the fallback is
post-training quantisation with percentile-99.9 calibration on up to
200 natural training images (MinMax if a tensor has no range).

Measured on Pascal VOC 2007 trainval (5011 images, 12,608 boxes, 20
classes, non-difficult; the sha1 split gives 4008 train / 1003 val) on
one RTX 3090, 25.6 minutes end to end including the compile:

| | AP@0.5 | note |
|---|---|---|
| float model | 0.542 | after 50 epochs |
| int8 export, PTQ (retired recipe) | 0.339 | every class hurt, -21 points |
| int8 export, QAT (shipped recipe) | **0.507** | -3.5 points |

Per class the QAT export is within 2 points of float for 13 of the 20
classes; the largest remaining gaps are sofa (-9.5), bicycle (-8.9),
train (-8.8), bus (-8.5) and pottedplant (-8.1). The compiled model:
0 pure-software epochs, 119 hardware + 3 hybrid, 923 kB of weights
(3.5% of the pool), 459 kB of activations, 274 MMACs.

A note on scale: these are numbers for a 0.9 M-parameter detector at
256x256 on a 20-class benchmark, which is a harder task than most
PixelKit datasets (a few classes, a fixed camera). On the 3-image QA
set and the 300-image VOC subset the QAT export matched or beat the
float model.

## Headless

`pixelkit train <dataset> --epochs 50 --input 256 --size nano --split 0.8
[--eval-dataset NAME] [--from RUN] [--resume RUN] [--device cuda:1] [--out DIR]`
runs the same job from a terminal and copies the export out. The
endpoints are in [API.md](API.md).

## Recording on the board

The camera the model will run on is the best source of training
pictures. Neuro Studio's Record button saves the board's stream as an
`.mjpeg` file; drop it on a dataset like a video and PixelKit splits it
into its frames (no conversion needed), with a trim and a "1 in N
frames" sample to keep near-identical frames out. Frames where nothing
is in view are good background images.

## Importing an existing dataset

Pascal VOC (`Annotations/*.xml`), COCO (any JSON with `images` and
`annotations`; polygon masks kept, RLE masks become boxes) and YOLO
(`labels/*.txt` with `classes.txt`, `obj.names` or `data.yaml`;
YOLO-seg polygons kept) all import as fully editable projects, from a
folder or a `.zip`.
