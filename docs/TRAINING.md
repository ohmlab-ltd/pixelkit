# Training and Neuro N6 export

PixelKit trains a YOLOX-Nano detector on a dataset and exports it for the
OhmLab Neuro N6 (STM32N657, Neural-ART NPU). Everything runs locally.

## What you get

`<dataset>/models/yolox_nano/<run>/`:

| File | Purpose |
|---|---|
| `<slug>_int8.onnx` | the deployable model: full-integer QDQ int8, raw head |
| `<slug>_labels.h` | `NUM_CLASSES`, `INPUT_W/H`, thresholds, `CLASSES[]` |
| `<slug>_snippet.ino.txt` | the `#pragma neuron6` line and decoder wiring to paste |
| `report.json` | AP@0.5 float vs int8 (overall and per class), input contract, NPU build facts |
| `export.zip` | the four files above, ready to drop into a sketch folder |
| `neuron6/` | ST Edge AI compiler log and generate report |
| `<slug>_fp32.onnx`, `best.pt` | the float model and checkpoint, for re-export or inspection |

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
weights pool, measured against 25.5 MB (the mempool declares 26.5 MB but
the last 1 MB is OTA app slot B). A model the compiler rejects is not
exported; the reason is recorded and shown in the app. Without the
compiler the export still succeeds and says `verified: false`.

## The recipe, validated on Pascal VOC 2007

Defaults: 50 float epochs plus 8 quantisation-aware epochs, 256x256,
batch 16 (32 on a 24 GB card is fine), SGD with warmup and cosine decay,
hflip + HSV + scale jitter, COCO warm start. With `qat_epochs=0` the
fallback is post-training quantisation with percentile-99.9 calibration
on up to 200 natural training images.

<!-- VOC_RESULTS -->

## Importing an existing dataset

Pascal VOC (`Annotations/*.xml`), COCO (any JSON with `images` and
`annotations`; polygon masks kept, RLE masks become boxes) and YOLO
(`labels/*.txt` with `classes.txt`, `obj.names` or `data.yaml`;
YOLO-seg polygons kept) all import as fully editable projects, from a
folder or a `.zip`.
