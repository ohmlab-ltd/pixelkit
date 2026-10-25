# Changelog

This changelog starts at 0.3.0.

## 0.4.0

Exports for Neuro N6 core 0.1.3 that letterbox the camera picture as
training does and can stream over WiFi, a checked and pinned Python
runtime, Electron 43, and fixes throughout the app.

### Added

- Openverse imports keep each image's title, creator and licence, and an
  export with Openverse images includes `ATTRIBUTIONS.txt`, which credits
  them, their crops in derived datasets and their augmented copies.
- An exported sketch can stream over WiFi: with `STREAM_OVER_WIFI` set to 1,
  a board with the MAYA-W476 radio hosts its own network for Neuro Studio and
  prints its name and password on the Serial Monitor.
- Help > Third-Party Notices opens the notices of the components PixelKit
  ships.
- Closing or reloading the window while an export is being built asks
  first.

### Changed

- An exported sketch pads the camera picture to a square with grey bars,
  `ASPECT_LETTERBOX`, as training does, instead of cropping its centre
  square.
- Training also shows the model the centre square of 30% of pictures that
  are not square, and varies the scale of background images as well as
  labelled ones.
- The exported sketch opens with a comment that names what it detects, the
  hardware it needs and the model it runs.
- The video picker no longer offers AVI, which the app cannot decode.
- Every exported model carries `NOTICE.txt` and `LICENSE.txt`: it is a
  modified YOLOX network under the Apache License, and the notice says when
  its weights descend from Megvii's COCO checkpoint, also through the runs
  it started from. A model trained from scratch had no notice.
- The installers carry PixelKit's LICENSE and NOTICE, the new
  THIRD-PARTY-NOTICES.md, YOLOX's licence and the font licence; the macOS app
  also keeps
  Electron's and Chromium's notices and the licences of its Python
  runtime, now a pinned python-build-standalone release.
- HEIC photos are decoded with pi-heif, the decode-only build of
  pillow-heif, whose wheels carry no GPL code. Pillow decodes AVIF itself.
- The NPU check builds the Neuro N6 memory map from the ST Edge AI Core
  install it compiles with, as the Neuro N6 core does, instead of shipping
  a copy of ST's file.
- Background images for augmentations upload as picked; the page shrank
  them to 1024 px and as little as 50 KB first.
- The Augmentations tab drops the Environmental effects and Object overlay
  rows, which could not be turned on, and a Project page drops Copy link,
  whose link opened only on the same computer.
- The engine drops the routes the app does not call: `POST /api/projects`,
  `GET /api/projects/{id}`, `GET /api/v3/projects/{id}/viewport`,
  `GET /api/jobs/{id}/events`, `DELETE /api/projects/{id}/jobs/{job}`,
  `POST /api/charlie/imports/process`, the originals and object-overlay
  routes, `GET /api/settings/hf-token`, and model download and unload. The
  routes in docs/API.md are unchanged. Dataset cards and Projects no longer
  return fields of the hosted service, such as `owner`, `private` or
  `members`, that older files still carry.

### Fixed

- When ST Edge AI crashes, cannot start or runs past 15 minutes, the export
  is marked unverified instead of rejected. A rejected run keeps its scores
  in its report. Closing PixelKit stops a compile in progress; on macOS it
  kept running.
- Neuro Studio `.mjpeg` recordings up to 2 GB import; the limit was the
  100 MB video limit.
- Renaming a label renames it everywhere: in every box, reference,
  augmentation and export, and in the text SAM 3 is prompted with. It used to
  change only the label's display name. Undo renames it back.
- Labels pasted or typed as a list, such as "pothole, crack", become
  separate labels.
- On Windows, a training run started while uploads were finishing could
  fail with "dataset not found". Reads now wait out a concurrent save, and a
  manifest that could not be read is never saved back over the dataset.
- Blurhash placeholders of imported images are filled in when missing.
- The overview shows one health score; the header and the health panel
  disagreed. The Train card's pre-flight refreshes after labelling, marking
  backgrounds and adding images.
- Cropped datasets keep each box on its object when crops are resized.
- The lens distortion augmentation no longer folds the picture back on
  itself above strength 3. It is a barrel distortion that keeps the corners
  in place.
- `python engine/import_legacy.py` given a single dataset folder imports that
  dataset only, not the folders beside it.
- The desktop app waits for an engine that is busy at launch instead of
  starting a second one. On Windows, an update no longer reinstalls the
  Python runtime when only comments in its requirements change.
- Turning camera augmentations off resets all nine of their dials.
- Training reads the boxes the exports read: detections the labeller
  rejected no longer train or count as validation ground truth.
- A resumed run keeps its train and validation split. Images added since
  train; the split used to be drawn again from the changed dataset.
- The COCO checkpoint's first layer gets its colour channels in the order it
  was trained on; runs started from it with red and blue swapped.
- A COCO checkpoint download cut short is no longer kept, which made every
  later run train from scratch. The file's SHA-256 is checked.
- A run that fails while packing its export stays resumable instead of
  listing as done with a broken export.
- Starting from an earlier run with different labels keeps the class
  weights of the labels both runs have.
- A label with a control character no longer breaks the labels header.
- The pre-flight lists boxes whose label is not one of the dataset's labels;
  they were left out silently.
- With several ST Edge AI Core versions installed, the NPU check uses the one
  the Neuro N6 core builds with, and compares versions by number.
- Datasets created under another user name, such as those from File >
  Import Legacy Backup, appear in the workspace and can join a Project.
  Every dataset on the computer belongs to the local user.
- An undo right after an edit could be lost on disk when a training
  pre-flight read the dataset at the same moment.
- Generating, clearing or deleting augmentations is refused while a
  training run on the dataset is queued or running, since training reads
  them every epoch. Starting or resuming a training run is refused while
  augmentations are being generated.
- Detections the labeller rejected are no longer copied into augmented
  copies.
- Settings saved at the same moment no longer overwrite each other, and the
  settings file is private to the user from the moment it is written.
- Model download progress counts each byte once.
- Augmented copies made before an image's boxes or masks were edited are
  left out of training and exports, so they no longer teach the old boxes.
  The Train card's pre-flight says how many images need Update on the
  Augmentations tab. Renaming or removing a label keeps copies current.
  Copies made by 0.3.0 carry no record of their boxes and are still used
  until Update remakes them.
- Exports list augmented copies of background images in their annotations.
  COCO, VOC, CVAT, Label Studio and mask exports shipped those images but
  left them out.
- Label Studio exports with polygons import: the labelling config has a
  polygon control for them.
- Restoring a snapshot or deleting a reference no longer leaves the
  dataset's cover naming a deleted image, and a cover left that way earlier
  moves to an existing image when it is shown.
- The macOS app can export models: its runtime lacked ONNX and ONNX
  Runtime.
- `pixelkit --help` keeps its usage table, and `pixelkit doctor` marks its
  checks `ok` and `!!`, which stay readable when its output is redirected.
- Starring a dataset works; the engine had no route for it, so every star
  was undone.
- Images dropped on a dataset upload 40 to a request. Every batch was
  refused, so each image went up on its own.
- A Hugging Face token saved with Compute set to CPU downloads SAM 3 at
  once; it waited for the next start.
- Adding more than 100 Openverse images to a dataset at once works; the
  engine refused the whole request.
- Renaming a dataset by editing its name at the top of the dataset page
  saves the name; it changed only the open page, and a reload brought back
  the old one. Ctrl+Z renames it back, and the title bar follows a rename
  made there or in the dataset's settings.
- In Review mode, the key of the verdict an image already has keeps that
  verdict. It cleared it, so going through the Good deck returned every
  image to Unrated.
- Opening a dataset no longer erases the masks of its reference images: the
  page wrote back reference boxes it had loaded without them.
- Opening a dataset no longer saves its labels unchanged, which moved it to
  the top of Recently updated.
- The dataset header shows the date the dataset last changed; it showed
  today's.
- A labelling or label-removal card ends when the engine restarts under it,
  instead of showing progress until the page is reloaded.
- Randomise on the Augmentations tab picks a block occlusion size of 5 to
  25%; it picked less than 1%.
- The All and Random choice shows only on the Camera category, the one
  category where the generator reads it.
- A dataset with a trained model shows the Model badge in the workspace and
  on its Project page; only datasets imported from a server showed it.
- Dataset covers on a Project page follow the dataset; they could stay as
  they were for a day.
- A dataset just created, starred or changed shows that way in the workspace
  list at once; the list could repeat its previous state for 2 seconds.
- An upload is one row in a Project's activity, with its image count; each
  image had its own row, which pushed every other event out.
- Crops deleted from a derived dataset in bulk, or as duplicates, stay
  deleted when it syncs with its parent; only a crop deleted on its own
  did.
- The gallery's All, Unlabelled and verdict counts follow uploads and
  deletes; they kept the earlier numbers until the next labelling run.
- A dataset with no images shows no health score, rating, coverage or
  duplicate verdict; it showed 100, Healthy and Full coverage.
- When the engine refuses to start labelling, remove a label or clear the
  annotations, the dataset page says why, and a label it refused to remove
  stays in the list.
- A changed Project cover shows at once in the Projects list instead of the
  browser's cached picture.
- A Project's activity calls a training run "trained a model".
- Reset in dataset settings replaces an uploaded cover; the upload stayed.
- A deleted dataset leaves the workspace list at once; it stayed, named by
  its id, until the engine restarted.
- A quick star and unstar, or two quick edits, no longer leave a card or the
  dataset page showing the earlier state.
- Augmentation progress counts up for the whole run instead of reading 0.
- The workspace search matches dataset names, labels and label aliases; any
  part of "local" matched every dataset.
- On Windows, a labelled preview no longer sometimes shows the original
  image right after an edit.

### Security

- The engine refuses changes sent by pages on other sites. A page open in
  the browser could post forms to it, for example to start a model
  download or create datasets.
- The engine answers only requests addressed to 127.0.0.1 or localhost, so
  a web page cannot reach it by pointing its own domain at this computer.
- Openverse image downloads connect only to the public address the check
  approved, so a host name that resolves to a private address on a second
  lookup is not fetched. They refuse a redirect to plain HTTP or to an
  address that is not public, as they refuse such a first address.
- A reference image name that points outside its dataset is refused.
- Uploads over 100 MB are refused on every route, with no more than that
  read into memory; some routes read uploads of any size.
- A reference image is stored only when it decodes, under its format's
  extension rather than its file name's, and responses carry
  `X-Content-Type-Options: nosniff`.
- Run checkpoints load with PyTorch's weights-only loader, which runs no
  pickled code.
- A legacy backup that would not fit on the disk is refused before it is
  unpacked.
- The unused upload route that stored files under the extension of their
  names, without decoding them, is removed.
- The Python runtime is checked against its published SHA-256 before it is
  unpacked, on the Windows first run and in the macOS build. A download that
  does not match is deleted and setup stops.
- The desktop app runs on Electron 43.7.9 instead of Electron 33, which no
  longer receives security fixes. The Windows installer grows to about
  115 MB.
- The engine no longer has the `VLM_WORKER_URL` setting, which sent whole
  images to a remote classifier when a label was ambiguous. Labelling runs
  on this computer only.
- Every package of the desktop app's Python runtime is pinned and checked
  against its SHA-256 as pip installs it. The Windows first run installs
  PyTorch 2.13.0 and the engine's packages from lock files instead of the
  newest releases, and the macOS build installs from its own lock, with the
  same versions.

## 0.3.0

Training of detectors for the Ohm Lab Neuro N6 from any labelled dataset.

### Added

- Training for the Neuro N6. The Train card trains YOLOX-Nano or YOLOX-Tiny,
  with square inputs from 128 to 416 px, and exports an Arduino sketch
  folder: the int8 model, trained quantisation-aware, a labels header with a
  confidence threshold calibrated on the validation images, and a report.
  Validation uses a per-class split or a second dataset. Generated
  augmentations join the training set. A run resumes after an interruption
  or starts from an earlier run. With ST Edge AI Core installed, each export
  is compiled for the NPU and its fit reported.
- The sketch streams its boxes to Neuro Studio, where the threshold is a live
  slider. It needs the Neuro N6 core 0.1.3 or later. On a Neuro N6, Nano runs
  in 7.4 ms at 256x256 and Tiny in 31 ms at 320x320, and the boxes match
  PixelKit's int8 predictions to within a few pixels.
- Per-class precision and recall, a confusion matrix, a gallery of the worst
  validation images, and every validation image beside its ground truth.
- Label images, on a finished run, labels the dataset's unlabelled images
  with its model. Images edited by hand are not changed.
- Background images: pictures with nothing to label count as labelled and
  train as negatives.
- Neuro Studio `.mjpeg` recordings import as frames, trimmed and sampled.
- COCO and YOLO datasets import as editable datasets, as Pascal VOC ones
  did.
- `pixelkit train` and `pixelkit label` on the command line, and the local
  API behind the Train card, described in docs/API.md.

### Fixed

- On a machine with one GPU, SAM 3 clicks and labelling jobs no longer wait
  for a training run; the run pauses between steps for them.
- Box-to-mask and click-to-detect find the object they point at; a faint
  whole-picture mask filtered it out. In tests they found 99% of objects, up
  from 55%.
- Augmented copies keep their boxes on the objects through rotation,
  perspective and lens distortion, and inside the picture.
- The editor keeps edits and masks, and a painted mask keeps its size through
  save and reopen.
- Fixes to imports, exports and storage that lost or reverted data.
- The desktop app keeps a busy engine running, cleans up on quit, and updates
  its Python runtime when a version needs new packages.
- A finished training run releases its GPU memory, and DINOv2 runs in half
  precision on any GPU chosen in Settings.

### Notes

- The installers are unsigned; SmartScreen and Gatekeeper warn on first
  launch.
- An exported model is fine-tuned from Megvii's COCO-pretrained YOLOX
  checkpoint unless its run trained from scratch. Its `NOTICE.txt` carries
  Megvii's Apache-2.0 notice, which stays with the model when it is
  distributed.
