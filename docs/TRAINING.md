# Training and Neuro N6 export

PixelKit trains a YOLOX detector, Nano or Tiny, on a dataset and exports it
for the Ohm Lab Neuro N6 (STM32N657, Neural-ART NPU). Training runs locally.
The local API behind the Train card is in [API.md](API.md).

## Run folder

`<dataset>/models/<yolox_nano|yolox_tiny>/<run>/`:

| File | Contents |
|---|---|
| `<slug>_int8.onnx` | the model: full-integer QDQ int8, raw head |
| `<slug>_labels.h` | `<SLUG>_NUM_CLASSES`, `<SLUG>_INPUT_W/H`, the calibrated `<SLUG>_CONF_THRESHOLD`, `<SLUG>_NMS_THRESHOLD`, `<SLUG>_CLASSES[]` |
| `<slug>.ino` | the sketch: the camera, the model at its input size, and the core's `PostProcess_PixelKitYOLOX`, which sends the boxes to Neuro Studio |
| `report.json` | AP@0.5 float and int8, overall and per class; threshold sweep; per-class precision and recall; confusion matrix; per-epoch history; input contract; NPU build summary |
| `export.zip` | a `<slug>/` sketch folder with the sketch, the int8 model, the labels header, `report.json`, `NOTICE.txt` and `LICENSE.txt` |
| `NOTICE.txt`, `LICENSE.txt` | the model is a modified YOLOX under the Apache License 2.0; the notice says when its weights descend from Megvii's COCO checkpoint |
| `failures/`, `failures.json` | the validation images with the most errors, with misses, spurious boxes and wrong classes drawn |
| `predictions.json` | each validation image's ground truth and the int8 model's boxes at the shipped threshold, in image pixels |
| `history.json` | per-epoch loss and AP@0.5, written during training |
| `neuron6/` | ST Edge AI compiler log and report |
| `<slug>_fp32.onnx`, `best.pt` | the float model and its checkpoint |
| `ckpt_last.pt` | present while a run is unfinished; Resume continues from it |

## Building for the board

The sketch folder builds in the Arduino IDE or with arduino-cli. It needs:

- The Neuro N6 core, 0.1.3 or later. Add
  `https://raw.githubusercontent.com/ohmlab-ltd/Neuro-N6-Arduino-Core/main/package_ohmlab_index.json`
  under File > Preferences > Additional boards manager URLs, then install
  Neuro N6 in the Boards Manager. The sketch checks the version and stops the
  build on an older core or another board.
- ST Edge AI Core, which the core runs to compile the model during the build.
  It is found as the core finds it: under `C:\ST\STEdgeAI\<version>` or
  `C:\Program Files\STMicroelectronics\STEdgeAI\<version>` on Windows,
  `/Applications/STEdgeAI/<version>` or `~/STEdgeAI/<version>` on macOS,
  `/opt/ST/STEdgeAI/<version>` or `~/STEdgeAI/<version>` on Linux, or on the
  PATH. `NEURON6_STEDGEAI` set to the `stedgeai` executable overrides the
  search.

Select the Neuro N6 board, open `<slug>.ino` and upload. The sketch pads the
camera picture to a square with grey bars (`ASPECT_LETTERBOX`), as in
training. The boxes stream to Neuro Studio over USB. On the TFT carrier,
`OV5640 camera(CARRIER_TFT);` and the `TRANSPORT_LCD` transport draw them on
the display instead.

### WiFi

On a board with the MAYA-W476 radio, `#define STREAM_OVER_WIFI 1` at the top
of the sketch streams over WiFi instead. The board hosts its own network,
with a password unique to the board. In Neuro Studio, connect the board over
USB first, then pick WiFi (the board's network) in the connection menu in the
status bar: Neuro Studio shows the network's name and password, and you join
that network on this computer. The Serial Monitor also prints them every 10
seconds. The network name is the
board's hostname: `neuron6-` and six hex digits, unless a hostname is stored
on the board from Neuro Studio's WiFi settings. The sketch's network takes
precedence over a network stored there.

## Training data

The training set is every labelled image in the train split, plus that
image's generated augmentations from the Augmentations tab, with their
transformed boxes. Augmentations of validation images are not used.

Background images train as negatives: pictures of the scene with nothing to
detect. An image with no boxes is marked background in the viewer, or in
bulk from the gallery's Select mode. On import, images whose annotation lists
no objects are marked background. A background image counts as labelled,
labelling jobs skip it, and drawing a box on it unmarks it. An image with no
boxes that is not marked background is unlabelled and is not used.

Training reads each image's boxes as the exports do: its edited boxes once
edited, else the detections the labeller did not reject.

### Split

The split is per class. Each class with two or more images gets at least one
validation image and keeps at least one training image, and is held out at
the chosen ratio, 60 to 90% of its images in train. Background images are
split as a class of their own, so the validation score counts boxes found
where there is nothing. The split is keyed on filenames and is the same on
every run. An image that also shows a class with a single image stays in
train; the pre-flight check names any class left unmeasured that way.

A second dataset can be the held-out set instead: every image in this dataset
trains, and every labelled image in the other one is scored, with labels
matched by name.

### Pre-flight

Before a run starts, a pre-flight check reports what will be used and what
is wrong. Blockers: too few labelled images, or background images without a
boxed image. Warnings: classes with fewer than ten boxes, classes the
validation split cannot measure, objects under 8 px at the input size,
byte-identical images on both sides of the split, boxes whose label is not
one of the dataset's labels, and images whose augmented copies were made
before their boxes changed. Warnings are recorded in the report.

## Nano or Tiny

| | Parameters | int8 weights | MACs at 256x256 | Use |
|---|---|---|---|---|
| YOLOX-Nano | 0.9 M | 0.9 MB | 0.27 G | default |
| YOLOX-Tiny | 5.1 M | 5 MB | 1.3 G | higher accuracy, under 20% of the weights pool |

Both use the same stem, head, quantisation and export. On the Neuro N6 NPU,
Nano runs in 7.4 ms at 256x256 and 5.4 ms at 192x192, and Tiny in 31 ms at
320x320. The boxes match PixelKit's int8 predictions on the same pictures to
within a few pixels.

Start from begins a run from a previous run's weights instead of the COCO
checkpoint. When the two runs' labels differ, each label both have keeps its
class weights, matched by name.

The COCO checkpoint is downloaded once into `weights/yolox/` and checked
against its SHA-256; a download cut short is not kept.

## Batch size and interruptions

The batch size comes from free VRAM at the start: about 70 MB per image for
Nano and 150 MB for Tiny at 256x256, within 60% of free memory, at most 64.
On Metal and the CPU the batch is at most 8. It is then reduced so that an
epoch has at least 15 optimiser steps; a 240-image dataset trains at batch
16. If the GPU runs out of memory, the batch
is halved and the epoch restarts, and the report records it.

A checkpoint is written after every float epoch. An interrupted run, from a
closed app, a crash or Cancel training, shows as Interrupted and resumes from
its last epoch with its own settings and its own split: images added since
train, and images deleted since are dropped. An interruption during the int8
fine-tuning restarts that phase from the float result. A run cancelled before
its first epoch completes shows as failed.

With two GPUs, training uses the one labelling is not using. With one,
training pauses between steps for SAM 3 clicks and labelling jobs. The Train
card plots loss and AP@0.5 during training, with the time remaining.

## Input contract

The core's build passes ST Edge AI `--input-data-type uint8
--inputs-ch-position chlast`: the camera writes packed RGB888 bytes straight
into the model's input buffer. This requires the graph input to be quantised
asymmetrically over [0,1], with scale 1/255 and zero-point -128. PixelKit
trains on [0,1] tensors, the QAT graph fixes the input fake-quant to those
parameters, and the export checks the file it wrote. A model whose input
range reaches outside [0,1] is refused, not rewritten; a narrower range is
set to the contract.

## Head

The graph has one output, the raw head: `N x (5 + classes)` rows of
`[dx, dy, dw, dh, obj, cls...]`. The grid decode runs on the Cortex-M55 in the
core's `yolox_pp.c`, because YOLOX's in-graph decode exports `ScatterND`,
`Expand` and `Where`, which ST Edge AI cannot shape-infer. The core's
`PostProcess_PixelKitYOLOX` wraps that decoder. The Focus stem is replaced by
an equivalent 6x6 stride-2 convolution, with the COCO stem weights remapped,
so that no `Slice` runs in software. The checkpoint was trained on BGR
pictures; the remap reverses its colour channels for PixelKit's RGB input.

## Quantisation-aware training

After the float epochs, PixelKit fine-tunes for 8 epochs with int8 simulated
at every tensor between ops: weights per channel, activations per tensor with
moving-average ranges, BatchNorm folded, the input fixed to the contract.
Post-training quantisation alone loses accuracy in the depthwise
backbone, where outlier channels set each activation tensor's range. The QDQ
graph is exported from that model, with int8 weights, int32 biases and Q/DQ
around every op, which ST Edge AI maps entirely to hardware. The number of
these epochs is `qat_epochs` in `TrainConfig`; the app, the API and the
command line keep it at 8. With `qat_epochs=0` the export uses post-training
quantisation instead, calibrated at the 99.9th percentile on up to 200
training images, or by MinMax when that calibration fails.

## NPU verification

With ST Edge AI Core installed where the core finds it, each export is
compiled as the core's `#pragma neuron6` build compiles it, against the
board's memory map. The result is in
`report.json`:

```json
"neuron6": {
  "verified": true,
  "epochsTotal": 122, "epochsSoftware": 0, "epochsHybrid": 3, "epochsHardware": 119,
  "weightsBytes": 919744, "poolUsedPct": 3.44, "poolRemainingBytes": 25818944,
  "activationsBytes": 458752, "macc": 271599330
}
```

`epochsSoftware` counts the epochs that run in software; 0 means the whole
network runs on the NPU. `poolRemainingBytes` is the space left in the
external-flash weights pool that the core's memory map declares: 25.5 MB,
ending where OTA slot B begins. PixelKit builds that map as the core does,
from the `stm32n6.mpool` of the ST Edge AI Core install it compiles with, and
a test compares the result with an installed core's map. A model the
compiler rejects is not exported; the run keeps its scores and the app shows
the reason. When the compiler is missing, cannot start, crashes, runs past 15
minutes or is stopped as PixelKit closes, the export succeeds with
`verified: false` and the reason.

## Threshold and errors

The decoder on the board uses one confidence threshold. After the int8 export
is scored, PixelKit sweeps the threshold from 0.05 to 0.95 on the validation
predictions and writes the one with the best F1, all classes pooled, into the
labels header, with its precision and recall. When several thresholds tie,
the middle of the plateau is used.

The report also has per-class precision and recall at that threshold, a
confusion matrix with a background row and column for spurious boxes and
misses, and a gallery of the validation images with the most errors. In the
gallery, green is the truth, thin blue a correct detection, yellow a miss,
red a spurious box and orange the right place with the wrong class. Details
shows every validation image twice: its ground truth, and the int8 model's
boxes at the shipped threshold.

The exported sketch registers the threshold with the core's live tuning
(`n6_tune_float`), so Neuro Studio shows a confidence slider while the sketch
runs. The board does not save the slider's value; a chosen value goes into
`<SLUG>_CONF_THRESHOLD`.

## Labelling with a trained model

Label images, on a finished run, labels the dataset's unlabelled images with
the exported int8 model, on the CPU at its calibrated threshold. Images edited
by hand are not changed.

## Recipe

Defaults: 50 float epochs and 8 quantisation-aware epochs, 256x256 input,
batch size from free VRAM, SGD with warmup and cosine decay, horizontal flip,
HSV and scale jitter on all images including background images, the centre
square of 30% of pictures, and the COCO-pretrained checkpoint as the start.
The centre square is what a sketch in `ASPECT_CROP_CENTER` mode shows the
model.

Pascal VOC 2007 trainval: 5011 images, 12,608 non-difficult boxes, 20
classes, split by file name into 3963 train and 1048 validation images. Three
runs, each about 30 minutes on one RTX 3090 including the compile:

| | AP@0.5 |
|---|---|
| float | 0.546 to 0.561 |
| int8, QAT | 0.495 to 0.501 |

The int8 score varies by a few points between splits and runs. The compiled
model runs entirely on the NPU: 0 software epochs, 119 hardware and 3
hybrid.

## Command line

```bash
pixelkit train <dataset> --epochs 50 --input 256 --size nano --split 0.8 \
    [--eval-dataset NAME] [--from RUN] [--resume RUN] [--device cuda:1] [--out DIR]
```

runs the same job and copies the export to `--out`. The endpoints are in
[API.md](API.md).

## Recordings from the board

Neuro Studio's Record button saves the board's stream as an `.mjpeg` file.
Dropped on a dataset like a video, it is split into its frames, with a trim
and a "1 in N frames" sample. A recording may be up to 2 GB. Frames with
nothing in view can be marked as background images.

## Importing datasets

Pascal VOC (`Annotations/*.xml`), COCO (a JSON file with `images` and
`annotations`; polygon masks are kept, RLE masks become boxes) and YOLO
(`labels/*.txt` with `classes.txt`, `obj.names` or `data.yaml`; YOLO-seg
polygons are kept) import as editable datasets, from a folder or a `.zip`.
