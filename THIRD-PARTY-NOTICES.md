# Third-party notices

PixelKit is licensed under the Apache License, Version 2.0 (LICENSE). This
file lists the third-party code, data, fonts and artwork in this repository,
and the third-party components that the PixelKit installers redistribute,
with their licences and copyright notices. Licence texts not reproduced
here are referenced by path or URL. Names are transliterated to ASCII.

1. In this repository
2. User interface bundle (ui/out, in both installers)
3. Desktop shell (both installers)
4. Python runtime (macOS dmg only)
5. Source code for LGPL and MPL components
6. Acknowledgements required by licences
7. Not redistributed
8. Licence texts

## 1. In this repository

### YOLOX 0.3.0 (vendored code)

- Files: engine/gd/vendor/yolox/. boxes.py, compat.py, darknet.py and
  losses.py are unchanged from the yolox 0.3.0 sdist. network_blocks.py,
  yolo_head.py, yolo_pafpn.py and yolox.py are modified by Ohm Lab Ltd and
  say so. qat.py and __init__.py are PixelKit's own. _decode_raw() in
  engine/gd/train_yolox.py is a numpy port of YOLOXHead.decode_outputs.
- Both installers carry engine/gd/vendor/yolox/ and its LICENSE in the
  application's resources folder.
- Source: https://github.com/Megvii-BaseDetection/YOLOX
- Licence: Apache-2.0 (section 8, and engine/gd/vendor/yolox/LICENSE)
- Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

### simplify-js (adapted code)

- Files: ui/app/MaskPainter.tsx, functions simplify() and sqSegDist()
- Source: https://github.com/mourner/simplify-js
- Licence: BSD-2-Clause (section 8)
- Copyright (c) 2017, Vladimir Agafonkin

### Geist and Geist Mono 1.200 (fonts)

- Files: ui/app/fonts/GeistVF.woff and ui/app/fonts/GeistMonoVF.woff,
  shipped as ui/out/_next/static/media/*.woff
- Source: https://github.com/vercel/geist-font
- Licence: SIL Open Font License 1.1 (section 8)
- Copyright 2024 The Geist Project Authors (https://github.com/vercel/geist-font)

### LVIS v1 category names (data)

- Files: ui/lib/words.txt lines 1 to 837, and the same names in ui/lib/words.ts
- Source: LVIS, https://www.lvisdataset.org. Gupta, Dollar and Girshick,
  "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019.
- Licence: Creative Commons Attribution 4.0,
  https://creativecommons.org/licenses/by/4.0/
- Changes: underscores replaced by spaces, qualifiers in parentheses
  removed, duplicates and some names left out. Lines 838 onward of
  words.txt are PixelKit's own.

### Icon artwork (inline SVG paths in ui/app)

- Feather icons (refresh, settings, alert triangle, pen tool, download
  tray, trash, filter, list, image, corner down left), in
  ui/app/v2/AugmentationsCard.tsx, ui/app/ProjectPage.tsx,
  ui/app/v2/ProjectViewV2Stub.tsx, ui/app/shell/ActivityBar.tsx,
  ui/app/v2/OverviewPanel.tsx, ui/app/v2/ReferenceImageEditor.tsx,
  ui/app/GuideView.tsx, ui/app/HomeView.tsx, ui/app/BoxEditor.tsx and
  ui/app/CreateProjectModal.tsx.
  Licence: MIT. Copyright (c) 2013-2023 Cole Bemis.
- Lucide icons (cloud download, loader circle, search), in
  ui/app/v2/OpenverseInlinePanel.tsx, ui/app/v2/DatasetStatsCard.tsx,
  ui/app/HomeView.tsx, ui/app/v2/VideoFrameModal.tsx and
  ui/app/v2/ProjectViewV2Stub.tsx.
  Licence: ISC. Copyright (c) 2026 Lucide Icons and Contributors. Icons
  derived from Feather: MIT, Copyright (c) 2013-present Cole Bemis.
- Material Design icons (star), in ui/app/HomeView.tsx.
  Licence: Apache-2.0. Copyright Google LLC.

## 2. User interface bundle (ui/out, in both installers)

The bundle is minified and drops most licence comments, so the notices
for the code in it are here.

| Component | Version | Licence | Copyright |
|---|---|---|---|
| Next.js (client runtime) | 14.2.35 | MIT | Copyright (c) 2024 Vercel, Inc. |
| mitt, in the Next.js client runtime | the copy of 1.1.3 in Next.js 14.2.35 | MIT | Copyright (c) Jason Miller |
| Node.js url.format, in the Next.js client runtime | the copy in Next.js 14.2.35 | MIT | Copyright Joyent, Inc. and other Node contributors. |
| React and React DOM, as vendored in next/dist/compiled | 18.3.0-canary-178c267a4e-20241218 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| scheduler, as vendored in next/dist/compiled | 0.24.0-canary-178c267a4e-20241218 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| react-server-dom-webpack client, as vendored in next/dist/compiled | the copy in Next.js 14.2.35 | MIT | Copyright (c) Meta Platforms, Inc. and affiliates. |
| react, react-dom | 18.3.1 | MIT | Copyright (c) Facebook, Inc. and its affiliates. |
| scheduler | 0.23.2 | MIT | Copyright (c) Facebook, Inc. and its affiliates. |
| web-vitals, as vendored in Next.js | 3.0.0 | Apache-2.0 | Copyright 2020 Google LLC |
| core-js (in the legacy polyfill chunk) | 3.38.1 | MIT | Copyright (c) 2014-2024 Denis Pushkarev |
| whatwg-fetch, in the legacy polyfill chunk | 3.0.0, as vendored in Next.js 14.2.35 | MIT | Copyright (c) 2014-2016 GitHub, Inc. |
| object-assign, in the legacy polyfill chunk | as vendored in Next.js 14.2.35 | MIT | Copyright (c) Sindre Sorhus |
| process, browser shim | 0.11.10, as vendored in Next.js 14.2.35 | MIT | Copyright (c) 2013 Roman Shtylman |
| buffer, in the annotation worker chunk | 5.6.0, as vendored in Next.js 14.2.35 | MIT | Copyright (c) Feross Aboukhadijeh, and other contributors. |
| base64-js, inside buffer | as vendored in Next.js 14.2.35 | MIT | Copyright (c) 2014 Jameson Little |
| ieee754, inside buffer | as vendored in Next.js 14.2.35 | BSD-3-Clause | Copyright 2008 Fair Oaks Labs, Inc. |
| styled-jsx | 5.1.1 | MIT | Copyright (c) 2016-present Vercel, Inc. |
| @swc/helpers | 0.5.5 | Apache-2.0 | SWC project (DongYoon Kang); the licence file has no copyright line |
| fflate | 0.8.3 | MIT | Copyright (c) 2026 Arjun Barrett |
| blurhash | 2.0.5 | MIT, declared in package.json; the npm package has no licence file | Author omahlama, https://github.com/woltapp/blurhash |
| react-blurhash | 0.3.0 | MIT (declared in package.json; no licence file published) | Author nygardk, https://github.com/woltapp/react-blurhash |
| Tailwind CSS base styles (preflight) | 3.4.19 | MIT | Copyright (c) Tailwind Labs, Inc. Preflight: Copyright (c) Nicolas Gallagher, Jonathan Neal, Sindre Sorhus, Adam Wathan, Jonathan Reinink |

The bundle also contains the fonts, word list, icon artwork and
simplify-js code listed in section 1.

## 3. Desktop shell (both installers)

| Component | Version | Licence | Copyright |
|---|---|---|---|
| Electron | 43.7.9 | MIT | Copyright (c) Electron contributors. Copyright (c) 2013-2020 GitHub Inc. |
| Chromium and Node.js, inside Electron, with their third-party components (FFmpeg under LGPL-2.1, ANGLE, Dawn, the DirectX Shader Compiler as dxcompiler.dll, SwiftShader, the Vulkan loader, ICU and others) | as shipped in Electron 43.7.9 | BSD-3-Clause, MIT and others | See LICENSES.chromium.html next to the application |
| d3dcompiler_47.dll, in Electron's Windows build | 10.0.26100.7705 | Distributable Code under the Microsoft Windows SDK licence terms; not covered by LICENSES.chromium.html | (c) Microsoft Corporation. All rights reserved. |
| dxil.dll, in Electron's Windows build | 1.8.2502.11 | Microsoft's licence terms for the binaries of the DirectX Shader Compiler release it comes from (February 2025); not covered by LICENSES.chromium.html | (c) Microsoft Corporation. All rights reserved. |

Windows installer only, added by electron-builder 26.17.0, MIT,
Copyright (c) 2015 Loopline Systems:

| Component | Version | Licence | Copyright |
|---|---|---|---|
| NSIS installer | 3.0.4.1 | zlib/libpng | Copyright (C) 1999-2018 Contributors |
| nsis7z plug-in (7-Zip code) | 19.00 | LGPL-2.1-or-later | Copyright (c) 1999-2016 Igor Pavlov, Nik Medved, Marek Mizanin, Stuart Welch |
| StdUtils plug-in | 1.1.4.0 | LGPL-2.1-or-later | Copyright (C) 2004-2018 LoRd_MuldeR |
| UAC and WinShell plug-ins | from electron-builder-binaries nsis-resources 3.4.1 | see https://github.com/electron-userland/electron-builder-binaries | their authors |
| elevate.exe (resources/elevate.exe) | 1.0 | not stated in the binary | Copyright (C) 2007 Johannes Passing |

The macOS dmg window uses electron-builder's default background image, from
dmg-builder 26.17.0, under the same licence.

On Windows, Electron's LICENSE is installed as LICENSE.electron.txt and the
Chromium notices as LICENSES.chromium.html in the install folder. On macOS
both are in PixelKit.app/Contents/Resources.

## 4. Python runtime (macOS dmg only)

PixelKit.app/Contents/Resources/runtime/py holds a relocatable CPython and
the engine's dependencies. Each package's licence files are in its
dist-info folder in runtime/py/lib/python3.12/site-packages/, with these
exceptions: onnxruntime keeps LICENSE and ThirdPartyNotices.txt in
site-packages/onnxruntime/; flatbuffers and tokenizers ship no licence
file, so their licence is the Apache-2.0 text listed in section 8; the
libraries bundled in tokenizers and torchvision are listed in their rows.

### python-build-standalone 20260807, aarch64-apple-darwin, install_only

The CPython build and the libraries it links. The licence texts of CPython
and of the libraries below are in runtime/py/licenses/, which
desktop/scripts/prepare-runtime.sh copies from desktop/licenses/python/,
the licence files of the release's full archive. pip's licence and those
of the packages it vendors are in its dist-info.

| Component | Version | Licence | Copyright |
|---|---|---|---|
| CPython | 3.12.13 | PSF-2.0 | Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved |
| bzip2 | 1.0.8 | bzip2-1.0.6 | Copyright (C) 1996-2019 Julian R Seward |
| expat | 2.8.2 | MIT | Thai Open Source Software Center Ltd, Clark Cooper and the Expat maintainers |
| libffi | 3.4.6 | MIT | Anthony Green, Red Hat, Inc and others |
| mpdecimal | 4.0.0 | BSD-2-Clause | Stefan Krah |
| OpenSSL | 3.5.7 | Apache-2.0 | The OpenSSL Project Authors |
| SQLite | 3.53.1 | Public domain | |
| Tcl and Tk | 9.0.4 | TCL | The Regents of the University of California, Sun Microsystems, Inc., Scriptics Corporation, ActiveState Corporation and other parties |
| libuuid | 1.0.3 | BSD-3-Clause | Theodore Ts'o |
| xz (liblzma) | 5.8.3 | 0BSD | The XZ Utils authors |
| pip | as shipped by the release | MIT | Copyright (c) 2008-present The pip developers; vendored packages listed in its dist-info |

### Python packages (engine/requirements-mac.txt)

| Package | Version | Licence | Copyright |
|---|---|---|---|
| accelerate | 1.14.0 | Apache-2.0 | The Hugging Face team |
| annotated-doc | 0.0.5 | MIT | Copyright (c) 2025 Sebastian Ramirez |
| annotated-types | 0.8.0 | MIT | Copyright (c) 2022 the contributors |
| anyio | 4.15.1 | MIT | Copyright (c) 2018 Alex Gronholm |
| blurhash | 1.1.5 | MIT | Copyright (c) 2019 Lorenz Diener |
| certifi | 2026.10.19 | MPL-2.0 | Kenneth Reitz; certificate data from Mozilla |
| click | 8.5.0 | BSD-3-Clause | Copyright 2014 Pallets |
| fastapi | 0.141.1 | MIT | Copyright (c) 2018 Sebastian Ramirez |
| filelock | 4.2.4 | MIT | Bernat Gabor and contributors |
| flatbuffers | 25.12.19 | Apache-2.0 | Copyright 2014 Google Inc. |
| fsspec | 2026.9.0 | BSD-3-Clause | Copyright (c) 2018, Martin Durant |
| h11 | 0.16.0 | MIT | Copyright (c) 2016 Nathaniel J. Smith and other contributors |
| hf-xet | 1.7.0 | Apache-2.0 | Hugging Face; compiled Rust crates are listed in its dist-info SBOM, including colored and option-ext under MPL-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause | Copyright (c) 2020, Encode OSS Ltd |
| httpx | 0.28.1 | BSD-3-Clause | Copyright (c) 2019, Encode OSS Ltd |
| huggingface_hub | 1.27.0 | Apache-2.0 | Hugging Face, Inc. |
| idna | 3.20 | BSD-3-Clause | Copyright (c) 2013-2026, Kim Davies and contributors |
| Jinja2 | 3.1.6 | BSD-3-Clause | Copyright 2007 Pallets |
| markdown-it-py | 4.2.0 | MIT | Copyright (c) 2020 ExecutableBookProject. Copyright (c) 2014 Vitaly Puzrin, Alex Kocharin |
| MarkupSafe | 3.0.4 | BSD-3-Clause | Copyright 2010 Pallets |
| mdurl | 0.1.2 | MIT | Copyright (c) 2015 Vitaly Puzrin, Alex Kocharin. Copyright (c) 2021 Taneli Hukkinen |
| ml_dtypes | 0.6.0 | Apache-2.0; includes Eigen code under MPL-2.0 | The ml_dtypes authors |
| mpmath | 1.3.0 | BSD-3-Clause | Copyright (c) 2005-2021 Fredrik Johansson and mpmath contributors |
| networkx | 3.7 | BSD-3-Clause | Copyright (c) 2004-2025, NetworkX Developers |
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Copyright (c) 2005-2025, NumPy Developers |
| onnx | 1.22.0 | Apache-2.0 | ONNX Project Contributors; its NOTICE (abseil-cpp, protobuf, nanobind) is in its dist-info |
| onnxruntime | 1.29.0 | MIT | Copyright (c) Microsoft Corporation; onnxruntime/ThirdPartyNotices.txt lists bundled code, including Eigen under MPL-2.0 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (OpenCV), MIT (packaging); bundled libraries under LGPL-2.1, LGPL-3.0, MPL-2.0, BSD, MIT and others | OpenCV contributors. Copyright (c) Olli-Pekka Heinisuo. Bundled libraries: LICENSE-3RD-PARTY.txt in its dist-info |
| orjson | 3.11.9 | MPL-2.0 AND (Apache-2.0 OR MIT) | The orjson authors (ijl) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Copyright (c) Donald Stufft and individual contributors |
| pi-heif | 1.4.0 | BSD-3-Clause; bundled libheif and libde265 under LGPL-3.0 | Copyright (c) 2021-2023, Pillow-Heif contributors; bundled libraries listed in its dist-info |
| Pillow | 12.2.0 | MIT-CMU; bundled brotli, FreeType, HarfBuzz, Little CMS, libavif, dav1d, libjpeg-turbo, libpng, libwebp, OpenJPEG, LibTIFF, xz and zlib-ng under their own licences | Copyright (c) 1997-2011 by Secret Labs AB. Copyright (c) 1995-2011 by Fredrik Lundh and contributors. Copyright (c) 2010 by Jeffrey A. Clark and contributors |
| protobuf | 7.36.2 | BSD-3-Clause | Copyright 2008 Google Inc. |
| psutil | 7.2.2 | BSD-3-Clause | Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola |
| pydantic | 2.14.1 | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors |
| pydantic_core | 2.50.1 | MIT; compiled Rust crates listed in its dist-info SBOM | Copyright (c) 2022 Samuel Colvin |
| Pygments | 2.21.0 | BSD-2-Clause | Copyright (c) 2006-2022 by the respective authors |
| python-dotenv | 1.2.2 | BSD-3-Clause | Copyright (c) 2014, Saurabh Kumar; 2013, Ted Tieken; 2013, Jacob Kaplan-Moss |
| python-multipart | 0.0.32 | Apache-2.0 | Andrew Dunham |
| PyYAML | 6.0.3 | MIT | Copyright (c) 2017-2021 Ingy dot Net. Copyright (c) 2006-2016 Kirill Simonov |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | Copyright 2020 Matthew Barnett. Portions copyright (c) 1998-2001 by Secret Labs AB |
| rich | 15.0.0 | MIT | Copyright (c) 2020 Will McGugan |
| safetensors | 0.8.0 | Apache-2.0 | Hugging Face |
| setuptools | 84.2.1 | MIT; vendored packages in setuptools/_vendor/ under their own licences, with their dist-info | Python Packaging Authority |
| shellingham | 1.5.4 | ISC | Copyright (c) 2018, Tzu-ping Chung |
| starlette | 1.9.0 | BSD-3-Clause | Copyright (c) 2018, Encode OSS Ltd |
| sympy | 1.14.0 | BSD-3-Clause | Copyright (c) 2006-2023 SymPy Development Team |
| tokenizers | 0.22.2 | Apache-2.0; bundles Oniguruma under BSD-2-Clause and Rust crates under their own licences; the wheel ships no licence or notice file | Hugging Face. Oniguruma: Copyright (C) 2002-2018 K.Kosako |
| torch | 2.13.0 | BSD-3-Clause; bundled third-party code under Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT, listed in its dist-info | Copyright (c) 2016- Facebook, Inc (Adam Paszke) and the other holders in its LICENSE |
| torchvision | 0.28.0 | BSD-3-Clause; bundled image libraries such as libjpeg-turbo, libpng and libwebp under their own licences, not included in its wheel | Copyright (c) Soumith Chintala 2016 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | The tqdm developers |
| transformers | 5.15.0 | Apache-2.0 | Copyright 2018- The Hugging Face team. All rights reserved. |
| typer | 0.27.3 | MIT | Copyright (c) 2019 Sebastian Ramirez |
| typing-inspection | 0.4.4 | MIT | Copyright (c) Pydantic Services Inc. 2025 to present |
| typing_extensions | 4.16.0 | PSF-2.0 | Python Software Foundation |
| uvicorn | 0.52.2 | BSD-3-Clause | Copyright (c) 2017-present, Encode OSS Ltd |

## 5. Source code for LGPL and MPL components

The licences of these components require their source code to be available
to everyone who receives the binaries. Their sources:

Both installers:

- FFmpeg in Electron 43.7.9 (LGPL-2.1), as ffmpeg.dll in the Windows
  install folder and libffmpeg.dylib in the Electron framework inside
  PixelKit.app: https://github.com/electron/electron and Chromium's
  third_party/ffmpeg

macOS dmg:

- FFmpeg, libbluray, libgnutls, libhogweed, libintl, libmp3lame, libnettle,
  libp11, librtmp, libsoxr and libtasn1 (LGPL-2.1), libgmp, libidn2 and
  libunistring (LGPL-3.0), as built into opencv-python-headless 5.0.0.93,
  with their licence texts in LICENSE-3RD-PARTY.txt in its dist-info:
  https://github.com/opencv/opencv-python, https://ffmpeg.org
- libsrt (MPL-2.0), in opencv-python-headless: https://github.com/Haivision/srt
- libheif and libde265 (LGPL-3.0), in pi-heif:
  https://github.com/strukturag/libheif, https://github.com/strukturag/libde265
- Eigen (MPL-2.0), in onnxruntime and ml_dtypes: https://gitlab.com/libeigen/eigen
- colored 3.1.1 and option-ext 0.2.0 (MPL-2.0), in hf-xet:
  https://crates.io/crates/colored, https://crates.io/crates/option-ext
- orjson 3.11.9 (MPL-2.0 portions): https://github.com/ijl/orjson
- certifi, tqdm and the copy of certifi vendored in pip (MPL-2.0) ship as
  Python source in the runtime.

Windows installer:

- nsis7z 19.00 (LGPL-2.1-or-later): https://nsis.sourceforge.io/Nsis7z_plug-in,
  https://www.7-zip.org
- StdUtils 1.1.4.0 (LGPL-2.1-or-later): https://github.com/lordmulder/stdutils

Written offer: for at least three years after Ohm Lab Ltd last distributes a
version of PixelKit, Ohm Lab Ltd will give anyone who asks a complete,
machine-readable copy of the source code of the LGPL components above, as
distributed in that version, for no more than the cost of performing the
distribution. Email team@ohmlab.co.uk with the PixelKit version and the
components you want.

## 6. Acknowledgements required by licences

- This software is based in part on the work of the Independent JPEG Group.
- Portions of this software are copyright (c) The FreeType Project
  (www.freetype.org). All rights reserved. This software is based in part
  on the work of the FreeType Team.

## 7. Not redistributed

None of these is part of the installers, and each carries its own licence.
PixelKit downloads the first three from their publishers on the user's
machine. The user installs the last one.

- Windows first run: CPython 3.12.13 from python-build-standalone 20260807,
  x86_64-pc-windows-msvc, install_only, from GitHub, or the CPython 3.12
  build of the newest release when that file cannot be reached; torch 2.13.0
  and torchvision 0.28.0 from download.pytorch.org, as pinned in
  engine/requirements-win-torch-cuda.txt or engine/requirements-win-torch-cpu.txt,
  where the CUDA build includes NVIDIA libraries under NVIDIA's licence terms;
  the packages pinned in engine/requirements-win.txt, from PyPI.
- Model weights: facebook/sam3 (Meta's SAM License, accepted by each user on
  Hugging Face), facebook/dinov2-large (Apache-2.0), and Megvii's YOLOX nano
  and tiny COCO checkpoints from the YOLOX GitHub release 0.1.1rc0
  (Apache-2.0).
- Images imported from Openverse, each under the licence recorded with it.
- ST Edge AI Core and the Neuro N6 Arduino core, installed by the user for
  the Neuro N6 export, under their own licences.

## 8. Licence texts

- Apache-2.0: LICENSE in the repository, LICENSE.txt in the installers
- SIL Open Font License 1.1: ui/app/fonts/OFL.txt in the repository,
  licenses/Geist-OFL.txt in the installers
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode
- MPL-2.0: https://mozilla.org/MPL/2.0/
- LGPL-2.1: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
- LGPL-3.0: https://www.gnu.org/licenses/lgpl-3.0.txt, with
  https://www.gnu.org/licenses/gpl-3.0.txt
- Unicode-3.0: https://www.unicode.org/license.txt
- PSF-2.0, TCL, bzip2-1.0.6, Zlib, BSL-1.0, 0BSD, MIT-CMU and the texts of
  every Python package: in the macOS runtime, as described in section 4
- Chromium and its components: LICENSES.chromium.html

### MIT License

Applies to the components marked MIT above, each with its own copyright
line.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### ISC License

Applies to the components marked ISC above, each with its own copyright
line.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

### BSD 2-Clause License

Applies to the components marked BSD-2-Clause above, each with its own
copyright line.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### BSD 3-Clause License

Applies to the components marked BSD-3-Clause above, each with its own
copyright line.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1.  Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3.  Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
