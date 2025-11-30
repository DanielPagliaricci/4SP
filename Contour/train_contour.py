#!/usr/bin/env python3
"""
Fine-tune Mask2Former for semantic segmentation of a single object contour.

This script:
  - Loads dataset saved by preprocess_contours.py (X_data.npy, Y_data.npy)
  - Builds a PyTorch dataset and collator using Mask2FormerImageProcessor
  - Fine-tunes a pretrained Mask2Former model for 2 classes (background/object)

All variables are hard-coded below (no argparse).
"""

from __future__ import annotations

import os

# Set critical env vars BEFORE importing torch/transformers so MPS fallback is honored
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # allow CPU fallback for unsupported MPS ops
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers import TrainerCallback
import random

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # optional


# ==========================
# Hyperparameters / Settings
# ==========================

# Dataset paths (produced by preprocess_contours.py)
DATASET_DIR: str = os.path.dirname(os.path.abspath(__file__))
X_DATA_PATH: str = os.path.join(DATASET_DIR, "X_data.npy")
Y_DATA_PATH: str = os.path.join(DATASET_DIR, "Y_data.npy")

# Model checkpoint and label mapping
# See: Mask2Former docs - https://huggingface.co/docs/transformers/model_doc/mask2former
PRETRAINED_CHECKPOINT: str = "facebook/mask2former-swin-base-ade-semantic"
NUM_LABELS: int = 2
ID2LABEL: Dict[int, str] = {0: "background", 1: "object"}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# Training hyperparameters
OUTPUT_DIR: str = os.path.join(DATASET_DIR, "mask2former_contour_finetuned")
NUM_EPOCHS: int = 10
TRAIN_BATCH_SIZE: int = 1  # keep minimal to avoid OOM
EVAL_BATCH_SIZE: int = 1
LEARNING_RATE: float = 5e-5
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.1
LOGGING_STEPS: int = 10
EVAL_STEPS: int = 100
SAVE_STEPS: int = 200
SEED: int = 42
USE_MIXED_PRECISION: bool = False  # keep False for macOS/MPS compatibility
TRAIN_VAL_SPLIT: float = 0.9  # fraction for training

# Optional caps to limit dataset size for low RAM scenarios
MAX_TRAIN_SAMPLES: int | None = 10000#None  # e.g. 1000
MAX_VAL_SAMPLES: int | None = 2000#None    # e.g. 200

# Visualization controls
SHOW_DATA_SAMPLES: bool = True
NUM_VIS_SAMPLES: int = 12

# (Env vars are set at import time above to take effect before torch loads)

# ==========================
# Augmentation Settings (no geometry changes)
# ==========================

AUGMENT_ENABLE: bool = True
# Gaussian blur probability and sigma range
AUG_BLUR_PROB: float = 0.5
AUG_BLUR_SIGMA_RANGE = (0.2, 2.5)  # include mild to strong

# Darkening probability and brightness factor range (<1 = darker)
AUG_DARKEN_PROB: float = 0.5
AUG_DARKEN_FACTOR_RANGE = (0.6, 0.95)  # strong to mild darkening

# Show a 4x3 grid of augmentation variations for one sample image
SHOW_AUGMENTED_GRID: bool = True


def log(msg: str) -> None:
    print(msg, flush=True)


def format_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def print_memory(prefix: str = "") -> None:
    if psutil is None:
        return
    vm = psutil.virtual_memory()
    pmem = psutil.Process().memory_info().rss
    log(f"[MEM]{' ' + prefix if prefix else ''} | sys used/total: {format_gb(vm.used)}/{format_gb(vm.total)} | proc RSS: {format_gb(pmem)}")


class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(total_steps, 1)

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if state.global_step % max(1, args.logging_steps) == 0:
            mem_note = ""
            if psutil is not None:
                rss = psutil.Process().memory_info().rss
                mem_note = f" | RSS: {format_gb(rss)}"
            log(f"[PROGRESS] step {state.global_step}/{self.total_steps} | epoch {state.epoch:.2f}{mem_note}")


def print_device_info() -> None:
    try:
        mps_built = getattr(torch.backends.mps, "is_built", lambda: False)()
        mps_avail = getattr(torch.backends.mps, "is_available", lambda: False)()
        log(f"[DEVICE] torch {torch.__version__} | MPS built: {mps_built} | MPS available: {mps_avail} | Fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    except Exception as e:
        log(f"[DEVICE] Info error: {e}")


# ==========================
# Dataset and Collator
# ==========================

class ContourDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, apply_augmentations: bool = False):
        assert images.ndim == 4 and images.shape[-1] == 3, "X must be (N,H,W,3)"
        assert masks.ndim == 3, "Y must be (N,H,W)"
        assert images.shape[0] == masks.shape[0], "X and Y must have same length"
        self.images_pil: List[Image.Image] = [Image.fromarray(img.astype(np.uint8), mode="RGB") for img in images]
        # Masks are integer class maps (0 background, 1 object)
        self.segmentation_pil: List[Image.Image] = [Image.fromarray(mask.astype(np.uint8), mode="L") for mask in masks]
        self.apply_augmentations: bool = bool(apply_augmentations)

    def __len__(self) -> int:
        return len(self.images_pil)

    def __getitem__(self, idx: int) -> Dict[str, Image.Image]:
        img = self.images_pil[idx]
        seg = self.segmentation_pil[idx]
        if self.apply_augmentations and AUGMENT_ENABLE:
            img = apply_augmentations_to_image(img)
        return {"image": img, "segmentation": seg}


def make_collate_fn(image_processor: Mask2FormerImageProcessor):
    def collate_fn(examples: List[Dict[str, Image.Image]]):
        images = [ex["image"] for ex in examples]
        seg_maps = [ex["segmentation"] for ex in examples]
        # Processor converts segmentation map to list of binary masks and class labels internally
        batch = image_processor(images=images, segmentation_maps=seg_maps, return_tensors="pt")
        return batch

    return collate_fn


# ==========================
# Augmentation helpers
# ==========================

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_augmentations_to_image(img: Image.Image) -> Image.Image:
    """Apply non-geometric augmentations: blur and darkening.

    - Does not change size, rotation, or position
    """
    out = img
    # Blur
    if AUG_BLUR_PROB > 0.0 and random.random() < AUG_BLUR_PROB:
        s_lo, s_hi = AUG_BLUR_SIGMA_RANGE
        sigma = random.uniform(min(s_lo, s_hi), max(s_lo, s_hi))
        if sigma > 0.0:
            out = out.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    # Darken
    if AUG_DARKEN_PROB > 0.0 and random.random() < AUG_DARKEN_PROB:
        f_lo, f_hi = AUG_DARKEN_FACTOR_RANGE
        factor = random.uniform(min(f_lo, f_hi), max(f_lo, f_hi))
        factor = _clamp(factor, 0.0, 1.0)  # ensure darkening only
        out = ImageEnhance.Brightness(out).enhance(factor)
    return out


def visualize_augmentation_grid(sample_np: np.ndarray) -> None:
    """Show a 3x4 grid (rows x cols) of augmentation variations including extremes."""
    pil = Image.fromarray(np.asarray(sample_np, dtype=np.uint8), mode="RGB")

    def apply_fixed(p: Image.Image, blur_sigma: float | None, dark_factor: float | None) -> Image.Image:
        q = p
        if blur_sigma is not None and blur_sigma > 0.0:
            q = q.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        if dark_factor is not None:
            q = ImageEnhance.Brightness(q).enhance(_clamp(float(dark_factor), 0.0, 1.0))
        return q

    # Define variations
    variations: List[tuple[str, Image.Image]] = []
    # Row 1: original + increasing blur
    variations.append(("orig", pil))
    variations.append(("blur 0.8", apply_fixed(pil, 0.8, None)))
    variations.append(("blur 1.5", apply_fixed(pil, 1.5, None)))
    variations.append(("blur 2.5", apply_fixed(pil, 2.5, None)))  # extreme
    # Row 2: original + increasing darken
    variations.append(("orig", pil))
    variations.append(("dark 0.9", apply_fixed(pil, None, 0.9)))
    variations.append(("dark 0.75", apply_fixed(pil, None, 0.75)))
    variations.append(("dark 0.6", apply_fixed(pil, None, 0.6)))  # extreme
    # Row 3: combined
    variations.append(("0.5+0.9", apply_fixed(pil, 0.5, 0.9)))
    variations.append(("1.0+0.8", apply_fixed(pil, 1.0, 0.8)))
    variations.append(("1.5+0.7", apply_fixed(pil, 1.5, 0.7)))
    variations.append(("2.5+0.6", apply_fixed(pil, 2.5, 0.6)))  # extreme both

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows))
    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        title, im_pil = variations[i]
        ax.imshow(im_pil)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================
# Visualization
# ==========================

def visualize_samples(images_np: np.ndarray, masks_np: np.ndarray, num_samples: int = 3, title_prefix: str = "") -> None:
    count = min(num_samples, images_np.shape[0])
    if count == 0:
        log("[INFO] No samples to visualize.")
        return
    cols = 4
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).ravel() if rows * cols > 1 else [axes]
    for i in range(count):
        img = np.asarray(images_np[i], dtype=np.uint8)
        msk = np.asarray(masks_np[i], dtype=np.uint8)
        ax = axes[i]
        ax.imshow(img)
        # Show mask contour
        ax.contour(msk, levels=[0.5], colors=['red'], linewidths=2)
        ax.set_title(f"{title_prefix} idx={i}")
        ax.axis('off')
    for j in range(count, rows * cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


# ==========================
# Training Logic
# ==========================

def main() -> None:
    set_seed(SEED)
    print_device_info()

    if not (os.path.isfile(X_DATA_PATH) and os.path.isfile(Y_DATA_PATH)):
        raise FileNotFoundError(
            f"Expected dataset files at: {X_DATA_PATH} and {Y_DATA_PATH}. Run preprocess_contours.py first."
        )

    X = np.load(X_DATA_PATH, mmap_mode="r")  # (N,H,W,3) uint8
    Y = np.load(Y_DATA_PATH, mmap_mode="r")  # (N,H,W) uint8 mask with values {0,1}
    print(f"[INFO] Loaded X: {X.shape} {X.dtype} | Y: {Y.shape} {Y.dtype}")

    num_samples = X.shape[0]
    if num_samples == 0:
        raise ValueError("Loaded empty dataset. Please check preprocessing step.")

    # Train/val split
    split_index = int(num_samples * TRAIN_VAL_SPLIT)
    train_images, val_images = X[:split_index], X[split_index:]
    train_masks, val_masks = Y[:split_index], Y[split_index:]

    # Optionally cap dataset sizes to avoid OOM
    if MAX_TRAIN_SAMPLES is not None and train_images.shape[0] > MAX_TRAIN_SAMPLES:
        train_images = train_images[:MAX_TRAIN_SAMPLES]
        train_masks = train_masks[:MAX_TRAIN_SAMPLES]
        print(f"[INFO] Capped train samples to {MAX_TRAIN_SAMPLES}")
    if MAX_VAL_SAMPLES is not None and val_images.shape[0] > MAX_VAL_SAMPLES:
        val_images = val_images[:MAX_VAL_SAMPLES]
        val_masks = val_masks[:MAX_VAL_SAMPLES]
        print(f"[INFO] Capped val samples to {MAX_VAL_SAMPLES}")

    train_ds = ContourDataset(train_images, train_masks, apply_augmentations=True)
    val_ds = ContourDataset(val_images, val_masks) if val_images.size > 0 else None
    print(f"[INFO] Dataset split -> train: {len(train_ds)} | val: {len(val_ds) if val_ds is not None else 0}")

    # Optional visualization before training
    if SHOW_DATA_SAMPLES:
        try:
            visualize_samples(train_images, train_masks, NUM_VIS_SAMPLES, title_prefix="train")
        except Exception as vis_exc:
            print(f"[WARN] Visualization failed: {vis_exc}")

    # Augmentation grid visualization: show for one training sample
    if SHOW_AUGMENTED_GRID and train_images.shape[0] > 0:
        try:
            visualize_augmentation_grid(train_images[0])
        except Exception as aug_exc:
            print(f"[WARN] Augmentation visualization failed: {aug_exc}")

    # Load image processor and model
    print("[INFO] Loading image processor...")
    image_processor = Mask2FormerImageProcessor.from_pretrained(PRETRAINED_CHECKPOINT)
    collate_fn = make_collate_fn(image_processor)

    print("[INFO] Loading model checkpoint (this may take a while)...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # re-init classification head for 2 classes
        low_cpu_mem_usage=True,
    )
    # Reduce memory: disable cache if available. Gradient checkpointing is not supported by this model.
    try:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass
    print("[INFO] Model loaded.")

    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to=[],  # disable wandb etc.
        remove_unused_columns=False,  # important with custom collate_fn
        fp16=USE_MIXED_PRECISION,
        dataloader_pin_memory=False,
    )

    # Attach progress callback with rough total steps
    total_steps = max(1, (len(train_ds) * NUM_EPOCHS) // max(1, TRAIN_BATCH_SIZE))
    progress_cb = ProgressCallback(total_steps)

    # Prefer CUDA on Vast.ai, then MPS (macOS), else CPU
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends.mps, "is_available", lambda: False)():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")
    model.to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[progress_cb],
    )

    print("[INFO] Starting training...")
    trainer.train()

    # Save final model and processor
    trainer.save_model(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Training completed. Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
