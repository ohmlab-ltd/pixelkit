# Changelog

Earlier releases are described on
[GitHub Releases](https://github.com/ohmlab-ltd/pixelkit/releases).

## 0.3.0

Train a detector for the OhmLab Neuro N6 from any labelled dataset and run
it on the board.

### New

- **Training for the Neuro N6.** The Train card trains YOLOX-Nano or
  YOLOX-Tiny (128 to 416 px square inputs) and exports an Arduino sketch
  folder: the int8 model, trained quantisation-aware so it scores like the
  float model, a labels header with a confidence threshold calibrated on
  the validation images, and a report. Validation is a per-class split or
  a second dataset; generated augmentations join the training set; a run
  resumes after an interruption or starts from an earlier run. With ST
  Edge AI Core installed, every export is compiled for the NPU and its fit
  reported.
- **On the board.** The sketch streams its boxes to Neuro Studio, where
  the threshold is a live slider. It needs the Neuro N6 core 0.1.3 or
  later and says so on an older one. Measured on a Neuro N6: Nano runs in
  7.4 ms at 256x256 and Tiny in 31 ms at 320x320, and the boxes match
  PixelKit's int8 predictions on the same pictures to within a few pixels.
- **How the model fails.** Per-class precision and recall, a confusion
  matrix, a gallery of the worst validation images, and every validation
  image beside its ground truth.
- **Label with your own model.** A finished run pre-labels the dataset's
  unlabelled images; hand-edited images are never touched.
- **Background images.** Pictures with nothing to label count as
  labelled and train as negatives.
- **Neuro Studio recordings.** An `.mjpeg` recording from the board
  imports as its frames, trimmed and sampled.
- **More import formats.** COCO and YOLO datasets import as editable
  projects, as Pascal VOC did.
- **Headless.** `pixelkit train` and `pixelkit label` from a terminal,
  and the local API behind the Train card ([docs/API.md](docs/API.md)).

### Fixed

- On a machine with one GPU, SAM 3 clicks and labelling jobs no longer
  wait for a training run to finish: the run pauses between steps for
  them.
- Box-to-mask and click-to-detect find the object they are pointed at
  (99% of objects in our tests, from 55%): a faint whole-picture mask used
  to filter it out.
- Augmented copies keep their boxes on the objects through rotation,
  perspective and lens distortion, and keep every box inside the
  picture.
- The editor never loses edits or masks, and a painted mask keeps its
  size through save and reopen.
- Imports and exports keep what they were given, and storage cannot lose
  or revert data.
- The desktop app keeps a busy engine running, cleans up on quit, and
  updates its AI runtime when a new version needs new packages.
- A finished training run gives its GPU memory back, and DINOv2 runs in
  half precision on any GPU chosen in Settings.

### Notes

- The installers are still unsigned, so SmartScreen and Gatekeeper warn
  on first launch.
- An exported model is fine-tuned from Megvii's COCO-pretrained YOLOX
  checkpoint: its `NOTICE.txt` carries Megvii's Apache-2.0 notice, to keep
  with the model when you share it.
