# Contributing

## Development setup

```bash
# engine/.venv with PyTorch and the engine, and the UI build
installers/install.sh           # Windows: installers\install.ps1

# Desktop shell, which runs the engine from engine/.venv
cd desktop && npm ci && npm start
```

Python 3.11 or later. CUDA PyTorch is optional: every API that does not run
a model works on the CPU. The engine looks for `ui/out` when it starts, so
the UI is built first. The installer adds the `pixelkit` command to
`engine/.venv`, with `pixelkit doctor` for environment checks.

Environment variables: `PIXELKIT_WORKSPACE` sets the workspace folder,
`PIXELKIT_CONFIG_DIR` the config folder, `PK_DISABLE_MODELS=1` runs the
engine without models, `PK_DEVICE=cuda|cuda:N|mps|cpu` selects the device,
`PIXELKIT_UI_DIR` serves another UI build, `PIXELKIT_PYTHON` sets the Python
the desktop shell runs the engine with, and `PIXELKIT_NO_UPDATE_CHECK=1`
turns off the update check.

## Tests

```bash
cd engine && pip install pytest httpx && python -m pytest tests/
cd ui && npm test && npx tsc --noEmit && npm run lint
```

The engine tests never load a model. CI runs the engine tests, an import
check of the SAM 3 classes in `transformers`, and the UI build and tests on
every push to `main` and every pull request.

## Rules

- The workspace format is a public contract ([docs/WORKSPACE.md](docs/WORKSPACE.md)).
  A change to what is written on disk updates that document and adds a
  migration in the same pull request.
- Every dataset, annotation and export API works with `PK_DISABLE_MODELS=1`.
- UI code uses the design tokens in [docs/DESKTOP-UI.md](docs/DESKTOP-UI.md).
- The installers' Python packages are pinned in `engine/requirements-runtime.in`
  and `engine/requirements-torch.in`. A change to either regenerates the
  hashed locks with the uv command at the top of each (the macOS lock with
  `MACOSX_DEPLOYMENT_TARGET=14.0`). `engine/requirements.txt` holds the ranges
  that source installs and CI use.
- A bug fix comes with a test that fails without it.
- Comments and documentation are short, factual, in the present tense and in
  ASCII.

## Releases

Set the version in `desktop/package.json`, `engine/pyproject.toml`,
`PK_VERSION` in `engine/gd/server.py` and `VERSION_FALLBACK` in
`ui/app/shell/StatusBar.tsx`, rename the `## Unreleased` section of
`CHANGELOG.md` to `## X.Y.Z`, then push a `vX.Y.Z` tag. The release workflow
builds both installers and publishes them with that section, followed by
GitHub's generated notes, as the release notes.
