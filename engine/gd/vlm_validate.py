"""Qwen 2.5-VL 7B validation pass over MM-GD-L detections.

Loads the model directly via Hugging Face Transformers (4-bit NF4 quant
via bitsandbytes) and runs in-process — no separate Ollama daemon, no
HTTP, no shared-VRAM probing problems with PyTorch's allocator.

For each auto-labelled box we crop the region, send it to Qwen, and
parse a YES/NO answer. Failures (model unloaded, OOM, etc.) come back
as `match=true, confidence=0` so the auto-label pipeline keeps moving;
the frontend renders these as un-verified rather than rejected.

Manual boxes skip this entirely — they're presumed valid because the
user drew them on purpose.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from PIL import Image

VLM_MODEL = os.environ.get("VLM_MODEL", "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")

# Where downloaded weights are cached on disk so we don't re-fetch ~4 GB
# every backend restart. Defaults to a `models_cache` folder next to the
# project on this machine; override with HF_HOME or VLM_CACHE_DIR if you
# want them somewhere else (shared scratch volume, etc.).
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "models_cache"
VLM_CACHE_DIR = Path(os.environ.get("VLM_CACHE_DIR") or os.environ.get("HF_HOME") or _DEFAULT_CACHE)
VLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Hugging Face's libraries read these directly; setting them ensures
# `from_pretrained` calls below find the cache without per-call kwargs.
os.environ.setdefault("HF_HOME", str(VLM_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(VLM_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(VLM_CACHE_DIR / "transformers"))

# Padding as a fraction of the box's own dimensions — gives the VLM real
# scene context. A tight car-grille crop with no padding looks like an
# abstract pattern; with 30% padding it's clearly the front of a car.
CROP_PADDING_FRAC = 0.3
CROP_PADDING_MIN_PX = 16
# Smallest crop side after padding. Smaller crops get upscaled so the VLM
# always has at least 384px to work with.
CROP_MIN_SIZE = 384
# Largest crop side. A 4K image with a small box can produce 1500+px
# crops which (a) blow peak VRAM during generate and (b) force the
# allocator to keep handing out new big buffers every call. Cap at
# 896 — well above what Qwen 2.5-VL actually uses internally
# (~1024px after its own resize) and small enough that consecutive
# generates can reuse the same allocator slabs.
# Capped at 448 (was 896) — Qwen3-VL's vision tower tokenises
# at 14 px patches, so 896×896 = 64×64 = 4096 vision tokens just
# for the image. Per-call latency was dominated by prefill on
# those tokens. 448×448 = 32×32 = 1024 tokens is plenty for the
# YES/which-label classification we actually run, and cuts
# prefill time roughly 4×. Override via VLM_CROP_MAX_SIZE env.
CROP_MAX_SIZE = int(os.environ.get("VLM_CROP_MAX_SIZE", "448"))

# Module-level handles set by `set_vlm` at backend startup. validate_box
# looks them up from here so callers don't need to thread state around.
_MODEL = None
_PROCESSOR = None


def set_vlm(model, processor) -> None:
    global _MODEL, _PROCESSOR
    _MODEL = model
    _PROCESSOR = processor


def clear_vlm() -> None:
    """Drop module references so the GC can reclaim VRAM. Used by the
    train job's unload-inference-models step."""
    global _MODEL, _PROCESSOR
    _MODEL = None
    _PROCESSOR = None


def is_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def load_vlm(device: str = "cuda"):
    """Load the configured VLM onto the given CUDA device. Strict
    GPU-only — refuses CPU fallback.

    Default model: Qwen3-VL-8B-Instruct-AWQ (pre-quantised). AWQ
    weights ship at ~5 GB on disk, load directly without runtime
    bnb dequant overhead, and run ~10-15% faster at inference than
    bnb-4bit. Set VLM_MODEL=... to swap models; the quant mode is
    auto-detected from the model id (AWQ / GPTQ in the name = pre-
    quantised, skip bnb config) but can be forced via VLM_QUANT.

    Quant modes:
      - "auto" (default): AWQ/GPTQ in model id → no bnb; otherwise
        4-bit bnb (NF4 + double-quant + fp16 compute).
      - "awq" / "gptq": pre-quantised, plain fp16 load.
      - "4bit" / "8bit": bnb on top of fp16 weights.
      - "fp16": no quantisation (needs more VRAM).
    """
    # Kill-switch: when VLM_DISABLED is set, skip the load entirely so
    # the rest of the stack (V2 GD/SAM/embeddings, V3 SAM3) has full
    # VRAM to itself. Both vlm_classify and validate_box already short-
    # circuit when _MODEL is None, so the V2 endpoints continue to work
    # — they just produce no VLM output.
    if os.environ.get("VLM_DISABLED", "").lower() in ("1", "true", "yes", "on"):
        print("[vlm] VLM_DISABLED set — skipping VLM load (frees ~10 GB VRAM).")
        return None, None

    import torch
    from transformers import AutoProcessor

    dev = str(device)
    if not (dev == "cuda" or dev.startswith("cuda")):
        raise RuntimeError(f"VLM requires CUDA, got device={device!r}")

    raw_quant_mode = os.environ.get("VLM_QUANT", "auto").lower()
    model_id_lower = VLM_MODEL.lower()
    is_prequant = "-awq" in model_id_lower or "-gptq" in model_id_lower or "-int4" in model_id_lower
    if raw_quant_mode == "auto":
        # Pre-quantised checkpoints carry their own kernels via
        # autoawq / auto_gptq registered with transformers — we just
        # load them in fp16 and the dequant happens internally.
        quant_mode = "awq" if is_prequant else "4bit"
    else:
        quant_mode = raw_quant_mode

    # Pick the right model class. Qwen3-VL needs its dedicated
    # ForConditionalGeneration class on transformers ≥ 4.46. We try
    # the official import first, then fall back to AutoModelForVision2Seq
    # so older transformers releases still load Qwen2.5-VL via the
    # generic auto path.
    model_cls = None
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Cls
        model_cls = _Cls
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
            if "qwen2.5" in VLM_MODEL.lower() or "qwen2_5" in VLM_MODEL.lower():
                model_cls = _Cls
        except ImportError:
            pass
    if model_cls is None:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    load_kwargs: dict = {
        "device_map": {"": dev},
        "cache_dir": str(VLM_CACHE_DIR / "hub"),
    }
    if quant_mode == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant_mode == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # awq / gptq / fp16 — fp16 dtype, no extra config. The
        # pre-quantised kernels (autoawq / auto_gptq) plug
        # themselves into transformers at import time and take
        # over the matmul layers internally.
        load_kwargs["torch_dtype"] = torch.float16

    print(f"[vlm] loading {VLM_MODEL} via {model_cls.__name__} (quant={quant_mode})")
    model = model_cls.from_pretrained(VLM_MODEL, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        VLM_MODEL,
        cache_dir=str(VLM_CACHE_DIR / "hub"),
    )
    return model, processor


def _crop(image: Image.Image, box: list[float]) -> Image.Image:
    W, H = image.size
    x0, y0, x1, y1 = box
    pad_x = max(CROP_PADDING_MIN_PX, int((x1 - x0) * CROP_PADDING_FRAC))
    pad_y = max(CROP_PADDING_MIN_PX, int((y1 - y0) * CROP_PADDING_FRAC))
    x0 = int(max(0, x0 - pad_x))
    y0 = int(max(0, y0 - pad_y))
    x1 = int(min(W, x1 + pad_x))
    y1 = int(min(H, y1 + pad_y))
    if x1 <= x0 or y1 <= y0:
        return image.crop((0, 0, W, H))
    crop = image.crop((x0, y0, x1, y1))
    if min(crop.size) < CROP_MIN_SIZE:
        scale = CROP_MIN_SIZE / max(1, min(crop.size))
        crop = crop.resize(
            (int(crop.size[0] * scale), int(crop.size[1] * scale)),
            Image.BICUBIC,
        )
    if max(crop.size) > CROP_MAX_SIZE:
        scale = CROP_MAX_SIZE / max(crop.size)
        crop = crop.resize(
            (int(crop.size[0] * scale), int(crop.size[1] * scale)),
            Image.BICUBIC,
        )
    return crop


def _strip_score(label: str) -> str:
    return re.sub(r"\s*\([\d.]+\)\s*$", "", label).strip()


def _highlight_mask_region(
    crop: Image.Image,
    source_image: Image.Image,
    box: list[float],
    polygons: list,
    *,
    fg_brighten: float = 1.18,
    bg_darken: float = 0.45,
) -> Image.Image:
    """Brighten the masked object inside `crop` and darken the
    background, so the VLM's attention is steered to the SAM
    silhouette rather than to whatever else happens to be in the
    crop's padding region.

    Replaces the older _draw_mask_outline approach (which drew a
    yellow polygon outline). VLMs were spending tokens describing
    the highlighter rather than the object — modulating the source
    pixels directly conveys the same "this is what to focus on"
    signal without injecting an obviously synthetic UI element.

    Polygons arrive in SOURCE-IMAGE coordinates. The crop is what
    `_crop` produced (padded, possibly resized to CROP_MIN_SIZE /
    CROP_MAX_SIZE). We replicate `_crop`'s padding maths to compute
    the offset (pad_x0, pad_y0), then derive the resize scale from
    the natural padded-crop size vs the actual crop size — that gives
    the affine to map polygon points into crop space.

    Multipliers:
      * fg_brighten=1.18 raises the masked pixels' luminance by ~18%
        without driving everything to pure white (clamped at 255).
      * bg_darken=0.45 cuts background luminance to ~45% — still
        readable for context (so the VLM can place the object in
        scene) but unmistakably dimmer than the foreground.
    Both clamp to [0, 255] after the multiply.
    """
    import numpy as np

    W, H = source_image.size
    if len(box) != 4:
        return crop
    x0, y0, x1, y1 = box
    pad_x = max(CROP_PADDING_MIN_PX, int((x1 - x0) * CROP_PADDING_FRAC))
    pad_y = max(CROP_PADDING_MIN_PX, int((y1 - y0) * CROP_PADDING_FRAC))
    pad_x0 = int(max(0, x0 - pad_x))
    pad_y0 = int(max(0, y0 - pad_y))
    pad_x1 = int(min(W, x1 + pad_x))
    pad_y1 = int(min(H, y1 + pad_y))
    natural_w = pad_x1 - pad_x0
    natural_h = pad_y1 - pad_y0
    if natural_w <= 0 or natural_h <= 0:
        return crop

    crop_w, crop_h = crop.size
    scale_x = crop_w / float(natural_w)
    scale_y = crop_h / float(natural_h)

    try:
        import cv2
    except ImportError:
        return crop

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    drew = False
    for poly in polygons or []:
        if not poly or len(poly) < 3:
            continue
        try:
            pts = np.asarray(
                [
                    (
                        int(round((float(p[0]) - pad_x0) * scale_x)),
                        int(round((float(p[1]) - pad_y0) * scale_y)),
                    )
                    for p in poly
                ],
                dtype=np.int32,
            )
        except (TypeError, ValueError, IndexError):
            continue
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 255)
            drew = True

    if not drew or not mask.any():
        return crop

    # Tighten the silhouette: a small close removes pixel-level
    # holes and a single dilate keeps the boundary slightly outside
    # the polygon under-shoot so the highlighted region covers the
    # full object — same idea as the DINOv2 inpaint mask cleanup.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    arr = np.array(crop).astype(np.float32)
    fg = (mask > 0)[..., None]
    out = np.where(fg, arr * fg_brighten, arr * bg_darken)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# Back-compat alias so any caller that still imports the old name
# keeps working — the new highlight is a strict superset of the
# outline behaviour (more attention-steering, no UI artefacts).
_draw_mask_outline = _highlight_mask_region


def validate_box(image: Image.Image, box: list[float], label: str) -> dict:
    """Run the VLM on one cropped box. Always returns a dict — on any
    error we fall back to `match=true, confidence=0` so the pipeline
    keeps moving; the frontend shows it as un-verified, not rejected."""
    if _MODEL is None or _PROCESSOR is None:
        return {
            "match": True,
            "confidence": 0.0,
            "reason": "vlm not loaded",
            "model": VLM_MODEL,
            "source": "auto",
        }

    import torch

    crop = _crop(image, box)
    clean = _strip_score(label) or "the labelled object"

    # Tightened prompt: explicit single-word constraint. Qwen still
    # sometimes elaborates ("It is", "Yeah") but we catch those in the
    # parser below.
    primary_prompt = (
        f"Is the main subject of this image a {clean}, "
        f"part of a {clean}, or something semantically equivalent "
        f"(synonym, sub-type, or close visual match)? "
        "Answer with a single word: YES or NO."
    )
    # Simpler retry prompt for degenerate-output cases. Less context,
    # less prompt depth — gets the model out of stuck-token loops.
    retry_prompt = f"Does this image show a {clean}? YES or NO."

    def _generate(prompt: str) -> str | None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = outputs = new_tokens = None
        try:
            text = _PROCESSOR.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = _PROCESSOR(
                text=[text],
                images=[crop],
                return_tensors="pt",
            ).to(_MODEL.device)
            with torch.inference_mode():
                outputs = _MODEL.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )
            new_tokens = outputs[:, inputs.input_ids.shape[1]:]
            return _PROCESSOR.batch_decode(new_tokens, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"[vlm] generation error: {e}")
            return None
        finally:
            # Drop refs to large CUDA tensors and reclaim cached blocks.
            # Without this the KV cache + decoder activations from each
            # generate() linger across calls (Python GC is non-deterministic
            # and the allocator keeps the slabs), so a 24 GB 4090 OOMs
            # after the first request even though weights only cost ~12 GB.
            del inputs, outputs, new_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _is_garbage(text: str) -> bool:
        if len(text) < 6:
            return False
        most_common = max(set(text), key=text.count)
        return text.count(most_common) / len(text) >= 0.8 and not re.search(r"[a-z]", text)

    raw = _generate(primary_prompt)
    if raw is None:
        return {
            "match": True,
            "confidence": 0.0,
            "reason": "vlm error",
            "model": VLM_MODEL,
            "source": "auto",
        }
    response = raw.strip().lower()
    # If the primary prompt produced degenerate output, retry once with
    # a simpler prompt before giving up.
    if _is_garbage(response):
        print(f"[vlm] {clean!r} primary garbage ({response[:40]!r}) — retrying simpler prompt")
        retry_raw = _generate(retry_prompt)
        if retry_raw is not None:
            retry_response = retry_raw.strip().lower()
            if not _is_garbage(retry_response):
                response = retry_response
    print(f"[vlm] {clean!r} -> {response[:160]!r}")

    if not response:
        return {
            "match": True,
            "confidence": 0.0,
            "reason": "vlm gave no answer",
            "model": VLM_MODEL,
            "source": "auto",
        }
    # If we're still stuck on garbage output even after the retry, mark
    # as a low-confidence match so a badge still renders. Keeps the FE
    # consistent — every box shows SOME validation state.
    if _is_garbage(response):
        return {
            "match": True,
            "confidence": 0.4,
            "reason": f"vlm garbage output: {response[:60]!r}",
            "model": VLM_MODEL,
            "source": "auto",
        }
    # Broad affirmative / negative vocab. Qwen replies are mostly "yes"
    # or "no" but sometimes "yeah", "it is", "correct", etc. Same goes
    # the other way with "nope", "not really", "incorrect". Catching
    # these gets the badge to render in cases where we'd otherwise have
    # silently fallen through to confidence=0.
    yes_pat = r"\b(yes|yeah|yep|yup|correct|true|affirmative|indeed|right|it\s+is|that\s+is|it\s*'\s*s)\b"
    no_pat = r"\b(no|nope|not|false|incorrect|wrong|negative|isn|doesn|does\s+not|is\s+not|n\s*'\s*t)\b"
    has_yes = re.search(yes_pat, response) is not None
    has_no = re.search(no_pat, response) is not None
    if has_no and not has_yes:
        return {
            "match": False,
            "confidence": 0.85,
            "reason": response[:240],
            "model": VLM_MODEL,
            "source": "auto",
        }
    if has_yes and not has_no:
        return {
            "match": True,
            "confidence": 0.85,
            "reason": response[:240],
            "model": VLM_MODEL,
            "source": "auto",
        }
    if has_yes and has_no:
        # Mixed wording ("yes but..." / "no, actually yes") — first hit
        # wins, lower confidence so the user knows it's hedged.
        first_yes = re.search(yes_pat, response)
        first_no = re.search(no_pat, response)
        if first_yes and first_no:
            if first_yes.start() < first_no.start():
                return {"match": True, "confidence": 0.55, "reason": response[:240], "model": VLM_MODEL, "source": "auto"}
            return {"match": False, "confidence": 0.55, "reason": response[:240], "model": VLM_MODEL, "source": "auto"}
    # Truly ambiguous — default to "verified, low confidence" so the
    # badge still shows. Better to overclaim a match than render a
    # blank slot the user can't tell from a not-checked box.
    return {
        "match": True,
        "confidence": 0.4,
        "reason": response[:240],
        "model": VLM_MODEL,
        "source": "auto",
    }


def _vlm_classify_remote(
    image: Image.Image,
    box: list[float],
    options: list[str],
    mask_polygons: list | None = None,
) -> tuple[str | None, float | None]:
    """Forward a classify_box call to the VLM worker over HTTP.

    Encodes the source image as JPEG once per call (typical sizes ~30-100
    KB on LAN — negligible vs the per-box VLM compute). The worker does
    its own padded crop server-side, so we send the full source image
    rather than the crop, matching the local `vlm_classify` semantics.

    Returns (None, None) on any failure — the upstream callers already
    treat that as "VLM declined" and fall back to embedding match or
    GD label, so a worker outage degrades gracefully.
    """
    import io
    import json as _json
    import os as _os

    try:
        import requests  # noqa: F401  (transformers pulls it; pinning explicit)
    except ImportError:
        # Extremely unlikely (transformers depends on it) but bail with
        # a clear message rather than a NameError.
        print("[vlm-remote] requests not installed — falling back to local")
        return None, None
    import requests

    url = (_os.environ.get("VLM_WORKER_URL") or "").rstrip("/")
    if not url:
        return None, None  # caller should have checked, but be defensive

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    buf.seek(0)

    headers: dict[str, str] = {}
    token = (_os.environ.get("VLM_WORKER_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 60s default. Qwen3-VL-8B AWQ on the 3060 should generate ≤16
    # tokens in 1-3s, but cold-start (model still loading the first
    # KV-cache pages, vision encoder warming up) can balloon the
    # first call past 10s. The tiebreak only fires on ambiguous
    # detections so we'd rather absorb the latency than time out.
    timeout = float(_os.environ.get("VLM_WORKER_TIMEOUT", "60"))

    # Pass mask polygons through so the worker can draw the outline
    # using the same logic as the local path. Empty / null masks are
    # OK — worker treats absence as "no outline available".
    data = {"box": _json.dumps(list(box)), "labels": _json.dumps(list(options))}
    if mask_polygons:
        data["mask_polygons"] = _json.dumps(mask_polygons)

    # Detailed timing so we can pinpoint where the wall clock goes
    # when the user reports a hang. The total split is:
    #   pre   — main backend prep (image encode, JSON serialise)
    #   wait  — full HTTP round-trip (includes the worker's prep,
    #           prefill, decode, response serialise + transit)
    #   post  — main backend post-processing (parse JSON)
    # Compare `wait` here with the worker's own [vlm-classify] line
    # to separate network latency from worker compute.
    import time as _t
    t_pre = _t.perf_counter()
    image_bytes = buf.getvalue()
    pre_ms = (_t.perf_counter() - t_pre) * 1000.0
    try:
        t_send = _t.perf_counter()
        r = requests.post(
            f"{url}/classify_box",
            files={"image": ("crop.jpg", image_bytes, "image/jpeg")},
            data=data,
            headers=headers,
            timeout=timeout,
        )
        wait_ms = (_t.perf_counter() - t_send) * 1000.0
    except requests.exceptions.Timeout:
        print(f"[vlm-remote] worker timeout after {timeout}s — falling back to local")
        return None, None
    except Exception as e:
        print(f"[vlm-remote] worker request failed: {e} — falling back to local")
        return None, None

    if r.status_code != 200:
        print(f"[vlm-remote] worker returned http {r.status_code}: {r.text[:200]}")
        return None, None

    t_post = _t.perf_counter()
    try:
        d = r.json()
    except Exception:
        return None, None
    label = d.get("label")
    score = d.get("score")
    post_ms = (_t.perf_counter() - t_post) * 1000.0
    print(
        f"[vlm-remote] options={list(options)} -> {label!r} "
        f"img_bytes={len(image_bytes)} pre={pre_ms:.0f}ms "
        f"wait={wait_ms:.0f}ms post={post_ms:.0f}ms"
    )
    return (label if isinstance(label, str) and label else None,
            float(score) if isinstance(score, (int, float)) else None)


def vlm_classify(
    image: Image.Image,
    box: list[float],
    options: list[str],
    mask_polygons: list | None = None,
) -> tuple[str | None, float | None]:
    """Pick the best-matching label from `options` for the cropped region.
    Returns (label, confidence) or (None, None) if the VLM declined to
    pick.

    When `mask_polygons` is supplied (SAM polygons in source-image
    coords), the function draws a coloured outline of the detection
    region on the crop before passing it to the VLM. This is the
    single biggest accuracy lift for the "people in front of a road"
    class of failures — without the outline the VLM picks whichever
    object is most visually salient (the people), but with it the
    model can scope its answer to the highlighted region.

    Remote routing: when `VLM_WORKER_URL` is set, the call is forwarded
    to a separate worker box. The worker re-uses this function locally,
    so prompt + mask-outline improvements take effect on both paths.
    """
    if not options:
        return None, None

    import os as _os
    if _os.environ.get("VLM_WORKER_URL"):
        return _vlm_classify_remote(image, box, options, mask_polygons)

    if _MODEL is None or _PROCESSOR is None:
        return None, None

    import torch

    crop = _crop(image, box)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    clean_opts: list[str] = []
    for opt in options:
        o = (opt or "").strip()
        if o and o.lower() not in seen:
            seen.add(o.lower())
            clean_opts.append(o)
    if not clean_opts:
        return None, None

    # If we have a SAM mask, draw it onto the crop so the VLM can see
    # exactly which region is in question. The prompt also changes to
    # explicitly reference "the outlined object" when an outline was
    # drawn — clearer instruction than "the main object".
    has_outline = False
    if mask_polygons:
        try:
            crop = _highlight_mask_region(crop, image, box, mask_polygons)
            has_outline = True
        except Exception as e:
            print(f"[vlm-classify] mask-outline draw failed: {e}; falling back to plain crop")

    # Prompt is deliberately short. Long prompts make Qwen-VL spend
    # most of its time on prefill rather than generating the actual
    # answer. The model only needs to know the candidate labels and
    # that we want one of them back; the rest was instruction-tuning
    # boilerplate it doesn't need.
    options_str = " | ".join(clean_opts)
    if has_outline:
        prompt = f"Which is the highlighted object: {options_str}? Reply with one label or 'none'."
    else:
        prompt = f"Which is the centred object: {options_str}? Reply with one label or 'none'."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    import time as _t
    t_total = _t.perf_counter()
    inputs = outputs = new_tokens = None
    try:
        t0 = _t.perf_counter()
        text = _PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = _PROCESSOR(
            text=[text],
            images=[crop],
            return_tensors="pt",
        ).to(_MODEL.device)
        prep_ms = (_t.perf_counter() - t0) * 1000.0
        n_input = int(inputs.input_ids.shape[1])
        t1 = _t.perf_counter()
        with torch.inference_mode():
            outputs = _MODEL.generate(
                **inputs,
                # 8 is enough for any single label word — Qwen tokenises
                # most short class names into 1-3 tokens. Generating up
                # to 16 was wasted decoding when the model wanted to
                # stop after the label but had no EOS yet.
                max_new_tokens=8,
                do_sample=False,
            )
        gen_ms = (_t.perf_counter() - t1) * 1000.0
        new_tokens = outputs[:, inputs.input_ids.shape[1]:]
        n_new = int(new_tokens.shape[1])
        response = _PROCESSOR.batch_decode(new_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"[vlm-classify] error: {e}")
        return None, None
    finally:
        # See note in validate_box._generate finally block — release
        # CUDA tensors so KV cache + activations don't accumulate.
        del inputs, outputs, new_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    response = response.strip().lower()
    total_ms = (_t.perf_counter() - t_total) * 1000.0
    print(
        f"[vlm-classify] options={clean_opts} -> {response[:120]!r} "
        f"crop={crop.size} input_tok={n_input} new_tok={n_new} "
        f"prep={prep_ms:.0f}ms gen={gen_ms:.0f}ms total={total_ms:.0f}ms"
    )

    if "none" in response and not any(o.lower() in response for o in clean_opts):
        return None, None
    # Match the first option that appears in the response. Anchored on
    # word boundaries so "car" doesn't match "carpet" if both happen to
    # be in the option list.
    for opt in clean_opts:
        if re.search(rf"\b{re.escape(opt.lower())}\b", response):
            return opt, 0.85
    # Fall back to substring match for partial responses.
    for opt in clean_opts:
        if opt.lower() in response:
            return opt, 0.6
    return None, None


def manual_validation() -> dict:
    """Marker stamped on user-drawn boxes — no VLM call, presumed correct."""
    return {"match": True, "confidence": 1.0, "reason": "manual", "model": None, "source": "manual"}
