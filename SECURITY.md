# Security policy

## Reporting a vulnerability

Email team@ohmlab.co.uk with a description, the affected versions and steps to
reproduce. Do not open a public issue.

## Scope

The PixelKit engine, the app's interface and desktop shell, and the installers
attached to its releases.

## Limits

- The engine listens on 127.0.0.1 without authentication: any program on the
  computer can read and change the workspace through it. It refuses requests
  addressed to another host name, and changes sent by pages on other sites.
- On first run the Windows app downloads its Python runtime from GitHub and
  its packages from PyPI and download.pytorch.org. Their integrity rests on
  HTTPS and those servers.
- SAM 3 and DINOv2 weights come from Hugging Face over HTTPS. The YOLOX
  COCO checkpoint comes from Megvii's GitHub release and is checked against
  a SHA-256 in the source.
- Exported models and sketches are not signed.
- The update check reads the latest release's version from GitHub and opens
  its page; it downloads no installer.
