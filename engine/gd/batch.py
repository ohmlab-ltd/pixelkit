"""Run Grounding DINO on every image in a project folder.

Layout expected:
    projects/<name>/
        images/         input images (.jpg/.jpeg/.png/.webp/.bmp)
        outputs/        annotated images + per-image json land here
        config.json     optional; e.g. {"prompt": "a pothole.", "box_threshold": 0.2}

Usage:
    python gd/batch.py <name>
    python gd/batch.py <name> --prompt "a pothole." --box-threshold 0.2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

from run_groundingdino import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, draw, load_image, load_model, predict

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULTS = {
    "prompt": "a pothole. a car.",
    "box_threshold": 0.25,
    "text_threshold": 0.25,
    "nms_iou": 0.5,
}


def load_config(project_dir, overrides):
    cfg = dict(DEFAULTS)
    cfg_path = project_dir / "config.json"
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text()))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def boxes_norm_to_xyxy_pixels(boxes_norm, W, H):
    if len(boxes_norm) == 0:
        return []
    xyxy = boxes_norm.clone()
    xyxy[:, :2] = boxes_norm[:, :2] - boxes_norm[:, 2:] / 2
    xyxy[:, 2:] = boxes_norm[:, :2] + boxes_norm[:, 2:] / 2
    xyxy = xyxy * torch.tensor([W, H, W, H])
    return xyxy.tolist()


def parse_phrase(label):
    if "(" in label and label.endswith(")"):
        text, score = label.rsplit("(", 1)
        try:
            return text.strip(), float(score.rstrip(")"))
        except ValueError:
            pass
    return label, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="project folder name under projects/")
    parser.add_argument("--prompt")
    parser.add_argument("--box-threshold", type=float, dest="box_threshold")
    parser.add_argument("--text-threshold", type=float, dest="text_threshold")
    parser.add_argument("--nms-iou", type=float, dest="nms_iou")
    args = parser.parse_args()

    project_dir = PROJECTS_DIR / args.project
    images_dir = project_dir / "images"
    outputs_dir = project_dir / "outputs"
    if not images_dir.is_dir():
        sys.exit(f"missing folder: {images_dir}")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(project_dir, {
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
    })

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        sys.exit(f"no images in {images_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_model(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, device)
    print(f"Project: {args.project}  prompt: {cfg['prompt']!r}  images: {len(image_files)}")

    summary = {"project": args.project, "config": cfg, "results": []}
    for img_path in image_files:
        image_pil, image_tensor = load_image(str(img_path))
        boxes_norm, phrases = predict(
            model, image_tensor, cfg["prompt"],
            cfg["box_threshold"], cfg["text_threshold"], device,
            nms_iou=cfg["nms_iou"],
        )
        W, H = image_pil.size
        xyxy = boxes_norm_to_xyxy_pixels(boxes_norm, W, H)

        detections = []
        for box, phrase in zip(xyxy, phrases):
            label, score = parse_phrase(phrase)
            detections.append({
                "label": label,
                "score": score,
                "box_xyxy": [round(v, 1) for v in box],
            })

        result = {
            "image": img_path.name,
            "size": {"width": W, "height": H},
            "detections": detections,
        }

        out_img = outputs_dir / f"{img_path.stem}_annotated.jpg"
        out_json = outputs_dir / f"{img_path.stem}.json"
        draw(image_pil.copy(), boxes_norm, phrases).save(out_img)
        out_json.write_text(json.dumps(result, indent=2))
        summary["results"].append(result)
        print(f"  {img_path.name}: {len(detections)} detection(s) -> {out_img.name}")

    (outputs_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {outputs_dir}")


if __name__ == "__main__":
    main()
