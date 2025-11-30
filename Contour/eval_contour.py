#!/usr/bin/env python3
"""
Evaluate a trained Mask2Former model and visualize predictions.

This script:
  - Loads X_data.npy and Y_data.npy created by preprocess_contours.py
  - Loads the fine-tuned model from mask2former_contour_finetuned
  - Computes binary mIoU and Dice for the object contour over a validation subset
  - Visualizes a few samples: overlays predicted vs ground-truth contours over the image

All variables are hard-coded below.
"""

from __future__ import annotations

import os
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.ndimage import binary_erosion, binary_dilation

#############################
# CONFIGURATIONS #
#############################

DATASET_DIR: str = os.path.dirname(os.path.abspath(__file__))
X_DATA_PATH: str = os.path.join(DATASET_DIR, "X_data_eval.npy")
Y_DATA_PATH: str = os.path.join(DATASET_DIR, "Y_data_eval.npy")
MODEL_DIR: str = os.path.join(DATASET_DIR, "mask2former_contour_finetuned")

# Same split as training
TRAIN_VAL_SPLIT: float = 0.001

# Evaluation subset and batching
EVAL_NUM_SAMPLES: int | None = 1000  # None = all validation
EVAL_BATCH_SIZE: int = 2

# Visualization
SHOW_SAMPLES: bool = True
NUM_VIS_SAMPLES: int = 6

# Class mapping: background=0, object=1
OBJECT_CLASS_ID: int = 1

# Boundary metric settings
BOUNDARY_TOLERANCE_PX: int = 2   # matching tolerance (radius in pixels)
BOUNDARY_THICKNESS_PX: int = 1   # boundary extraction thickness


def log(msg: str) -> None:
    print(msg, flush=True)


def get_device() -> torch.device:
    if getattr(torch.backends.mps, "is_available", lambda: False)():
        return torch.device("mps")
    return torch.device("cpu")


def postprocess_to_binary_masks(
    image_processor: Mask2FormerImageProcessor,
    outputs: torch.Tensor | dict,
    batch_heights: List[int],
    batch_widths: List[int],
) -> List[np.ndarray]:
    target_sizes: List[Tuple[int, int]] = [(h, w) for h, w in zip(batch_heights, batch_widths)]
    semseg_maps = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=target_sizes)
    # Convert semantic class map to binary mask for the object class
    masks_bin: List[np.ndarray] = []
    for seg in semseg_maps:
        seg_np = seg.cpu().numpy().astype(np.int32)
        masks_bin.append((seg_np == OBJECT_CLASS_ID).astype(np.uint8))
    return masks_bin


def compute_confusion(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    assert pred.shape == gt.shape
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, fn


def compute_iou_dice(tp: int, fp: int, fn: int, eps: float = 1e-8) -> Tuple[float, float]:
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return iou, dice


def make_disk(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= (r*r)


def compute_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    mask_b = mask.astype(bool)
    if mask_b.sum() == 0:
        return np.zeros_like(mask_b)
    selem = make_disk(max(1, thickness))
    eroded = binary_erosion(mask_b, structure=selem, iterations=1, border_value=0)
    boundary = np.logical_and(mask_b, np.logical_not(eroded))
    return boundary


def boundary_fscore(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2, thickness: int = 1, eps: float = 1e-8) -> Tuple[float, float, float, int, int]:
    """Compute boundary Precision/Recall/F with tolerance.

    Returns (precision, recall, fscore, pred_boundary_count, gt_boundary_count)
    """
    pred_b = compute_boundary(pred, thickness)
    gt_b = compute_boundary(gt, thickness)

    pred_count = int(pred_b.sum())
    gt_count = int(gt_b.sum())

    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0, pred_count, gt_count

    tol_selem = make_disk(tolerance)
    gt_dil = binary_dilation(gt_b, structure=tol_selem)
    pred_dil = binary_dilation(pred_b, structure=tol_selem)

    pred_match = np.logical_and(pred_b, gt_dil).sum()
    gt_match = np.logical_and(gt_b, pred_dil).sum()

    precision = pred_match / (pred_count + eps) if pred_count > 0 else (1.0 if gt_count == 0 else 0.0)
    recall = gt_match / (gt_count + eps) if gt_count > 0 else (1.0 if pred_count == 0 else 0.0)
    f = (2 * precision * recall) / (precision + recall + eps)
    return float(precision), float(recall), float(f), pred_count, gt_count


def visualize_overlays(images_np: np.ndarray, gts_np: np.ndarray, preds_np: List[np.ndarray], num: int) -> None:
    n = min(num, images_np.shape[0], len(preds_np))
    if n <= 0:
        log("[INFO] No samples to visualize.")
        return
    fig, axes = plt.subplots(n, 1, figsize=(7, 5 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = np.asarray(images_np[i], dtype=np.uint8)
        gt = np.asarray(gts_np[i], dtype=np.uint8)
        pr = np.asarray(preds_np[i], dtype=np.uint8)
        ax = axes[i]
        ax.imshow(img)
        ax.contour(gt, levels=[0.5], colors=['lime'], linewidths=2)
        ax.contour(pr, levels=[0.5], colors=['red'], linewidths=2)
        ax.set_title(f"idx={i} | GT=green, Pred=red")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Load dataset
    if not (os.path.isfile(X_DATA_PATH) and os.path.isfile(Y_DATA_PATH)):
        raise FileNotFoundError("X_data.npy / Y_data.npy not found. Run preprocessing first.")

    X = np.load(X_DATA_PATH, mmap_mode="r")  # (N,H,W,3)
    Y = np.load(Y_DATA_PATH, mmap_mode="r")  # (N,H,W)
    total = X.shape[0]
    if total == 0:
        raise ValueError("Empty dataset.")
    log(f"[INFO] Loaded dataset: X={X.shape} {X.dtype} | Y={Y.shape} {Y.dtype}")

    # Train/val split
    split_idx = int(total * TRAIN_VAL_SPLIT)
    val_images = X[split_idx:]
    val_masks = Y[split_idx:]
    val_count = val_images.shape[0]
    log(f"[INFO] Validation size: {val_count}")
    if val_count == 0:
        raise ValueError("No validation samples. Increase validation split or provide more data.")

    # Subset
    if EVAL_NUM_SAMPLES is not None and val_count > EVAL_NUM_SAMPLES:
        val_images = val_images[:EVAL_NUM_SAMPLES]
        val_masks = val_masks[:EVAL_NUM_SAMPLES]
        log(f"[INFO] Using subset of validation: {val_images.shape[0]}")

    # Load model + processor
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    device = get_device()
    log(f"[DEVICE] Using device: {device}")
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_DIR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # Evaluate in batches
    num_samples = val_images.shape[0]
    batch_size = max(1, EVAL_BATCH_SIZE)
    tp_total = fp_total = fn_total = 0
    preds_all: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            imgs_pil: List[Image.Image] = [Image.fromarray(np.asarray(val_images[i], dtype=np.uint8), mode="RGB") for i in range(start, end)]
            heights = [img.size[1] for img in imgs_pil]
            widths = [img.size[0] for img in imgs_pil]

            enc = processor(images=imgs_pil, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            pred_bin_masks = postprocess_to_binary_masks(processor, outputs, heights, widths)

            # accumulate metrics
            for i_local, pred_mask in enumerate(pred_bin_masks):
                idx = start + i_local
                gt_mask = np.asarray(val_masks[idx], dtype=np.uint8)
                tp, fp, fn = compute_confusion(pred_mask, gt_mask)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                # boundary metrics (micro-aggregate via counts)
                p, r, f, pred_cnt, gt_cnt = boundary_fscore(
                    pred_mask, gt_mask, tolerance=BOUNDARY_TOLERANCE_PX, thickness=BOUNDARY_THICKNESS_PX
                )
                # accumulate via weighted P/R using counts (micro-average)
                # store per-sample P/R counts
                # sum of matched counts approximated by p*pred_cnt and r*gt_cnt
                # but we didn't compute matched counts directly; recompute using tolerance again is expensive.
                # Instead, aggregate P/R by averaging per-sample P/R (macro average):
                # We'll keep running sums of P and R and divide by N at the end.
                if 'p_sum' not in locals():
                    p_sum = 0.0
                    r_sum = 0.0
                    n_b = 0
                p_sum += p
                r_sum += r
                n_b += 1
            preds_all.extend(pred_bin_masks)

    miou, dice = compute_iou_dice(tp_total, fp_total, fn_total)
    log(f"[RESULT] mIoU: {miou:.4f} | Dice: {dice:.4f} (over {num_samples} validation samples)")

    if 'n_b' in locals() and n_b > 0:
        p_avg = p_sum / n_b
        r_avg = r_sum / n_b
        f_avg = (2 * p_avg * r_avg) / (p_avg + r_avg + 1e-8)
        log(
            f"[RESULT] Boundary-F (tolerance={BOUNDARY_TOLERANCE_PX}px, thickness={BOUNDARY_THICKNESS_PX}px): "
            f"Precision={p_avg:.4f} | Recall={r_avg:.4f} | F={f_avg:.4f}"
        )

    # Visualization
    if SHOW_SAMPLES:
        visualize_overlays(val_images, val_masks, preds_all, NUM_VIS_SAMPLES)


if __name__ == "__main__":
    main()


