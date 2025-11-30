#!/usr/bin/env python3
"""
Run inference with a fine-tuned Mask2Former model on all images in a folder.

- Loads processor and model from a local directory (default: mask2former_contour_finetuned)
- Converts grayscale images to RGB automatically
- Predicts a binary segmentation (background/object) and overlays the contour on the original image
- Saves results to an output directory alongside the test images
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Tuple

# Ensure deterministic tokenizer behavior and allow MPS fallback
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

#############################
# CONFIGURATIONS #
#############################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "mask2former_contour_finetuned")
OUTPUT_DIR = os.path.join(BASE_DIR, "Predictions")
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "Database", "Only_Obj_Env", "Rendered_Images")


def log(msg: str) -> None:
    print(msg, flush=True)


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if is_image_file(f)]
    return files


def ensure_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def load_model(model_dir: str) -> Tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, torch.device]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Train first or update MODEL_DIR."
        )
    log(f"[INFO] Loading processor from: {model_dir}")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    log(f"[INFO] Loading model from: {model_dir}")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()
    device = get_device()
    model.to(device)
    log(f"[DEVICE] torch {torch.__version__} | device: {device}")
    return processor, model, device


def predict_segmentation(
    pil_img: Image.Image,
    processor: Mask2FormerImageProcessor,
    model: Mask2FormerForUniversalSegmentation,
    device: torch.device,
) -> np.ndarray:
    pil_img = ensure_rgb(pil_img)
    inputs = processor(images=pil_img, return_tensors="pt")
    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Post-process to a HxW map of label ids (expects (height, width))
    height, width = pil_img.height, pil_img.width
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    seg_np = seg.cpu().numpy().astype(np.uint8)
    return seg_np


def overlay_and_save(
    pil_img: Image.Image,
    seg_map: np.ndarray,
    out_path: str,
    title: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pil_img)
    # Binary mask for the foreground/object class assumed id=1
    mask = (seg_map == 1)
    try:
        ax.contour(mask, levels=[0.5], colors=["red"], linewidths=2)
    except Exception:
        # Fallback: if contour fails (e.g., mask empty), show mask overlay with alpha
        ax.imshow(np.where(mask, 255, 0), cmap="Reds", alpha=0.4)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_object_only_image(
    pil_img: Image.Image,
    seg_map: np.ndarray,
) -> Image.Image:
    """Return a full-size RGB image where background is black and object retains color.

    The foreground/object class is assumed to have label id=1.
    """
    img_rgb = ensure_rgb(pil_img)
    img_np = np.array(img_rgb)
    mask = (seg_map == 1)
    if not mask.any():
        # No object detected: return fully black image of the same size
        return Image.fromarray(np.zeros_like(img_np), mode="RGB")
    masked_np = img_np * mask[..., None]
    return Image.fromarray(masked_np, mode="RGB")


def _compute_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Compute tight bounding box (x_min, y_min, x_max_exclusive, y_max_exclusive) for a boolean mask.

    Returns None if mask has no True pixels.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1  # exclusive
    x_min = int(xs.min())
    x_max = int(xs.max()) + 1  # exclusive
    return x_min, y_min, x_max, y_max


def crop_object_with_margin(
    pil_img: Image.Image,
    seg_map: np.ndarray,
    margin_ratio: float = 0.30,
) -> Image.Image | None:
    """Produce a SQUARE crop centered on the object with extra margin; background inside crop is black.

    - The square side is based on the object's largest dimension.
    - margin_ratio: fraction of the largest dimension to add on each side (default 0.25).
    - Returns None if no object pixels are present.
    """
    img_rgb = ensure_rgb(pil_img)
    img_np = np.array(img_rgb)
    mask = (seg_map == 1)

    bbox = _compute_bbox_from_mask(mask)
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    h, w = mask.shape
    obj_w = max(1, x_max - x_min)
    obj_h = max(1, y_max - y_min)

    # Square side based on the larger object dimension plus margin on each side
    largest_dim = max(obj_w, obj_h)
    desired_side = largest_dim * (1 + 2 * margin_ratio)
    side = int(round(max(1.0, desired_side)))
    # Ensure the square fits within the image
    side = min(side, min(w, h))

    # Center at the object's bbox center
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    half = side // 2
    # Start with an integer-centered square; adjust to fit in bounds
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    # Clamp so the square stays within the image
    x0 = max(0, min(w - side, x0))
    y0 = max(0, min(h - side, y0))
    x1 = x0 + side
    y1 = y0 + side

    # Crop image and mask
    crop_img_np = img_np[y1 - side:y1, x1 - side:x1]
    crop_mask = mask[y1 - side:y1, x1 - side:x1]

    if crop_mask.size == 0:
        return None

    crop_masked_np = crop_img_np * crop_mask[..., None]
    return Image.fromarray(crop_masked_np, mode="RGB")


def main(args: Iterable[str]) -> None:
    base_dir = BASE_DIR

    root_dir = ROOT_DIR
    
    # Paths (update if you want to use a different model or folder)
    model_dir = MODEL_DIR
    test_images_dir = TEST_IMAGES_DIR
    output_dir = OUTPUT_DIR

    # Allow optional overrides via CLI: model_dir test_images_dir output_dir
    argv = list(args)
    if len(argv) >= 1 and argv[0]:
        model_dir = argv[0]
    if len(argv) >= 2 and argv[1]:
        test_images_dir = argv[1]
    if len(argv) >= 3 and argv[2]:
        output_dir = argv[2]

    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"Test images folder not found: {test_images_dir}")

    processor, model, device = load_model(model_dir)
    image_paths = list_images(test_images_dir)
    if not image_paths:
        log(f"[WARN] No images found in: {test_images_dir}")
        return

    # Prepare sub-directories for additional outputs
    object_only_dir = os.path.join(output_dir, "ObjectOnly")
    cropped_dir = os.path.join(output_dir, "Cropped")
    os.makedirs(object_only_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    log(f"[INFO] Found {len(image_paths)} images. Starting inference...")
    for idx, img_path in enumerate(image_paths, start=1):
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            log(f"[ERROR] Failed to open {img_path}: {e}")
            continue

        pil_img = ensure_rgb(pil_img)
        seg_map = predict_segmentation(pil_img, processor, model, device)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_png = os.path.join(output_dir, f"{img_name}_contour.png")
        overlay_and_save(pil_img, seg_map, out_png, title=img_name)
        log(f"[OK] ({idx}/{len(image_paths)}) Saved: {out_png}")

        # Save full-size object-only image (background black)
        obj_img = create_object_only_image(pil_img, seg_map)
        obj_out = os.path.join(object_only_dir, f"{img_name}_object.png")
        obj_img.save(obj_out)
        log(f"[OK] ({idx}/{len(image_paths)}) Saved: {obj_out}")

        # Save cropped image around object with margin, background black
        crop_img = crop_object_with_margin(pil_img, seg_map, margin_ratio=0.25)
        if crop_img is not None:
            crop_out = os.path.join(cropped_dir, f"{img_name}.png")
            crop_img.save(crop_out)
            log(f"[OK] ({idx}/{len(image_paths)}) Saved: {crop_out}")
        else:
            log(f"[WARN] No object detected in {img_name}; skipping cropped output.")


if __name__ == "__main__":
    main(sys.argv[1:])


