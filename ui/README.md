# PixelKit UI

Next.js interface for PixelKit: workspace, annotation editor for boxes and
polygon masks with click-to-segment, review mode, augmentations, dataset
statistics, import and export, and training.

`npm run build` writes a static export to `out/`, which the engine serves.
`npm run dev` serves the UI on port 3000 and expects the engine at
`NEXT_PUBLIC_API_URL`, `http://localhost:8001` by default (`.env.example`).

Tests: `npm test`.
