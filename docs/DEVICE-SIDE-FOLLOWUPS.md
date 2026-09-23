# Follow-ups that need the Arduino core or NeuroStudio

PixelKit's side of "dataset in, deployable model out" is done up to the
export: training (background images included, as negatives),
quantisation, threshold calibration, NPU verification, the bundle,
pre-labelling with the trained model, every val image beside its ground
truth, headless training and the local API ([API.md](API.md)). On the
board, the exported sketch shows its confidence threshold as a live
slider in Neuro Studio (the core's `n6_tune_float`). The items below
close the remaining gaps to the competition but need changes on the
device side, so they are deliberately not started here.

## Needs the Arduino core

1. **On-device validation.** Run the val split on the board and compare
   its raw head rows with the host's int8 predictions, so the number in
   the report is the number on silicon. Needs a host-to-device bulk
   transport (NeuroStudio removed its FWU0 push) and a core mode that
   accepts frames over USB and returns the raw tensor. The host side
   is ready: `predictions.json` holds the int8 model's boxes for every
   val image, so the comparison is a small addition once frames can be
   sent.
2. **Anomaly detection (PatchCore-style).** The NPU runs the backbone,
   the M55 keeps the memory bank and scores patches; that memory-bank
   and scoring code is core work, and the weights-pool budget for the
   bank has to be decided there first.
3. **Keypoints and tracking.** Pose heads and a tracker are decoders in
   `PostProcess`, not PixelKit work.

## Needs NeuroStudio

4. **Fetch and flash a PixelKit export.** NeuroStudio can read
   `GET /api/v2/projects/{id}/models` and `.../export` from the local
   engine, list the runs with their AP and NPU facts, and build + flash
   the chosen one without the user moving a zip.
5. **Show the report next to the board.** Per-class precision/recall,
   the confusion matrix, the failure gallery and `predictions.json`
   already exist under the run folder; NeuroStudio can present them
   beside the live camera view.

## Unblocked on the device, open in PixelKit

6. **Classification models.** The core now runs any one-output
   classifier (`PostProcess_Classifier`: a softmax when the output is
   raw scores, the best classes sent as `META_IMAGE_CLASS`) and
   NeuroStudio shows them in its Image Classifier panel. What is left
   is PixelKit's side: a class per image (today a class can only be
   given by drawing a box), a MobileNet-class trainer on those or on
   derived crop datasets, and an export whose sketch binds
   `PostProcess_Classifier`. Quantisation, verification and the bundle
   carry over.
