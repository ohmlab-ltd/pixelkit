# Follow-ups that need the Arduino core or NeuroStudio

PixelKit's side of "dataset in, deployable model out" is done up to the
export: training, quantisation, threshold calibration, NPU verification,
the bundle, pre-labelling with the trained model, headless training and
the local API ([API.md](API.md)). The items below close the remaining
gaps to the competition but need changes on the device side, so they
are deliberately not started here.

## Needs the Arduino core

1. **On-device validation.** Run the val split on the board and compare
   its raw head rows with the host's int8 predictions, so the number in
   the report is the number on silicon. Needs a host-to-device bulk
   transport (NeuroStudio removed its FWU0 push) and a core mode that
   accepts frames over USB and returns the raw tensor. PixelKit has the
   per-image val predictions and the letterbox geometry in memory when
   it scores an export (it persists only the worst-image gallery), so
   the comparison is a small addition once frames can be sent.
2. **Classification models.** A MobileNet-class classifier needs a core
   post-processor (softmax + argmax over the labels header) and a demo
   sketch; PixelKit would train on derived datasets (one crop per box,
   the crop's label as the class). Same export path otherwise.
3. **Anomaly detection (PatchCore-style).** The NPU runs the backbone,
   the M55 keeps the memory bank and scores patches; that memory-bank
   and scoring code is core work, and the weights-pool budget for the
   bank has to be decided there first.
4. **Keypoints and tracking.** Pose heads and a tracker are decoders in
   `PostProcess`, not PixelKit work.

## Needs NeuroStudio

5. **Fetch and flash a PixelKit export.** NeuroStudio can read
   `GET /api/v2/projects/{id}/models` and `.../export` from the local
   engine, list the runs with their AP and NPU facts, and build + flash
   the chosen one without the user moving a zip.
6. **Threshold on the device.** The report carries the full
   precision/recall sweep; a slider in NeuroStudio that rewrites
   `_CONF_THRESHOLD` (or sends it at runtime) and shows the expected
   precision/recall would let the user tune on live frames.
7. **Show the report next to the board.** Per-class precision/recall,
   the confusion matrix and the failure gallery already exist as JSON
   and JPEGs under the run folder; NeuroStudio can present them beside
   the live camera view.
