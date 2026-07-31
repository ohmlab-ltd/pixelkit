"""Lightweight NSFW gate for uploads.

Uses NudeNet (YOLOv5-based ONNX detector). Locates specific anatomical regions
rather than classifying the whole frame, so it's robust to screenshots, UI
chrome, borders, and compression — the cases a binary classifier misses.

`nsfw_score(detector, path)` returns (score, class) for the highest-scoring
"exposed" detection, or (0.0, "") if none. Caller applies the threshold.
"""
from pathlib import Path
from nudenet import NudeDetector

EXPOSED_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
}

# Project image uploads also block COVERED variants — catches lingerie /
# swimwear / bikinis, which is fine for a pothole/PPE workflow but produces
# false positives on red round objects (tomatoes, fruit, etc.). Avatars use
# the stricter EXPOSED-only set instead.
BLOCK_CLASSES = EXPOSED_CLASSES | {
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_COVERED",
}


def load_classifier(device: str):
    # device kept for signature compatibility; NudeNet runs on CPU via ONNX.
    return NudeDetector()


def nsfw_score(detector, path: Path, classes: set[str] | None = None) -> tuple[float, str]:
    block = classes if classes is not None else BLOCK_CLASSES
    detections = detector.detect(str(path))
    best_score = 0.0
    best_class = ""
    for d in detections:
        cls = d.get("class", "")
        if cls in block:
            score = float(d.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_class = cls
    return best_score, best_class
