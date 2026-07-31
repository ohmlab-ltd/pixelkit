# PixelKit UI

Next.js workspace UI: annotation editor (boxes, polygon masks,
click-to-segment), review mode, augmentation designer, dataset stats,
YOLO/COCO/VOC import + export.

Dev: `npm install && npm run dev` (expects the engine on localhost:8001 —
see .env.example). Auth/billing/telemetry from the SaaS build are removed;
a static local session is provided in `app/SessionProviderWrapper.tsx`.
