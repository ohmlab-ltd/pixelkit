"""Edit the values below and run: python detect.py"""
import os

import torch

from run_groundingdino import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, draw, load_image, load_model, predict

# ---- edit these ----
IMAGE_PATH = "potholes.png"
PROMPT = "a pothole. a car."
OUTPUT_PATH = "potholes_output.jpg"
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
# --------------------

CONFIG = DEFAULT_CONFIG
CHECKPOINT = DEFAULT_CHECKPOINT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = IMAGE_PATH if os.path.isabs(IMAGE_PATH) else os.path.join(root, IMAGE_PATH)
    output_path = OUTPUT_PATH if os.path.isabs(OUTPUT_PATH) else os.path.join(root, OUTPUT_PATH)

    image_pil, image = load_image(image_path)
    model = load_model(CONFIG, CHECKPOINT, DEVICE)
    boxes, phrases = predict(model, image, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, DEVICE)
    print(f"Found {len(boxes)} detection(s):")
    for p in phrases:
        print(f"  - {p}")
    draw(image_pil, boxes, phrases).save(output_path)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
