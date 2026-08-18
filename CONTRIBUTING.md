# Contributing to PixelKit

Thanks for looking under the hood. The short version: small, tested,
honestly-described changes are easy to merge.

## Dev setup

```bash
# engine - Python 3.12; CUDA torch optional (CPU works for all non-ML APIs)
cd engine && pip install -r requirements.txt && python gd/server.py
# → http://127.0.0.1:8001

# ui - Next.js static export, served by the engine after a build
cd ui && npm ci && npm run build     # or `npm run dev` on :3000

# desktop shell (optional in dev)
cd desktop && npm ci && npm start
```

`pip install -e engine` gives you the `pixelkit` launcher and
`pixelkit doctor`. Useful env vars: `PIXELKIT_WORKSPACE` (isolate your
data), `PK_DISABLE_MODELS=1` (run with no ML), `PK_DEVICE=cpu|cuda:N`.

## Tests

```bash
cd engine && python -m pytest tests/     # fast; models are never loaded
cd ui && npx tsc --noEmit && npx eslint .
```

CI runs both on every push/PR, plus an import gate that catches
transformers releases breaking the SAM 3 classes.

## Ground rules

- **Workspace format is a public contract** (`docs/WORKSPACE.md`):
  changes to what's written on disk need a doc update and a migration
  story in the same PR.
- The engine must keep working with `PK_DISABLE_MODELS=1` - every
  dataset/annotation/export API stays model-free.
- Match the code around you (the UI's design tokens are documented in
  `docs/DESKTOP-UI.md`); keep comments for the *why*, not the what.
- Add a regression test when you fix a bug - the suite is small and
  fast on purpose.

## Releases

Maintainers: bump the version stamps in `desktop/package.json`,
`engine/gd/server.py` (`PK_VERSION`), `engine/pyproject.toml`, and
`ui/app/shell/StatusBar.tsx`, then push a `vX.Y.Z` tag - the release
workflow builds and publishes both installers.
