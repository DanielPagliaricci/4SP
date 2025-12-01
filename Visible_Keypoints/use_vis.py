#!/usr/bin/env python3
"""
Inference script for the TANGO visible-vertex heatmap + coordinate model.

Matches preprocessing used in `TANGO_preprocess_heatmap.py` and model
architecture/losses in `TANGO_train_heatmap.py`.

It loads the best checkpoint if available, otherwise the final saved model.
It iterates over image/JSON pairs (same folders as preprocessing), runs
predictions, prints per-vertex errors for visible vertices, and saves
visualizations (overlay + per-vertex heatmaps).
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from tensorflow.keras.models import load_model

# Add current directory to sys.path to allow importing sibling scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

try:
    import preprocess_vis as preproc
    from train_vis import configure_gpu, SoftArgMax, focal_bce, masked_l1
except ImportError as e:
    print(f"Error importing helper modules (preprocess_vis, train_vis): {e}")
    print("Make sure you are running this script from the Visible_Keypoints directory or that it is in your PYTHONPATH.")
    sys.exit(1)


#############################
# CONFIGURATIONS #
#############################
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
MODEL_CKPT_PATH = os.path.join(LOG_DIR, "best_model.keras")   # PREFER THIS IF IT EXISTS
MODEL_FINAL_PATH = os.path.join(SCRIPT_DIR, "visible_keypoints_model.keras")
PRIORITIZE_BEST_CHECKPOINT = True  # IF FALSE, TRY FINAL MODEL FIRST

##### INPUT DATA SOURCES #####
CUSTOM_IMG_DIR = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Rendered_Images")              # If set (and USE_PREPROC_DIRS=False), read images from here
CUSTOM_JSON_DIR = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "JSON_Data_REF")             # If set (and USE_PREPROC_DIRS=False), read JSONs from here

##### OUTPUTS #####
OUTPUT_VIS_DIR = os.path.join(SCRIPT_DIR, "predictions")

##### INFERENCE BEHAVIOR #####
VISUALIZE_FIRST_N = 5             # Visualize first N samples (set 0 to disable)
LIMIT_SAMPLES = None               # If set to int, process at most that many samples

##### OPTIONAL OVERRIDES #####
FORCE_NUM_V = None                 # e.g., 11 to force vertex count; None to auto
FORCE_IMG_SZ = None                # e.g., 128 to force display scale; None to use preproc
FORCE_HEAT_SZ = None               # e.g., 64; not strictly needed at inference

##### PREDICTION SCOPE #####
# If set, this directory is used as the source of images to predict,
# overriding either the preprocessing IMG_DIR or CUSTOM_IMG_DIR.
IMAGES_DIR_TO_PREDICT = None       # e.g., "/path/to/only_these_images"


##### PLOTTING CONTROL #####
PLOT_GROUND_TRUTH = True           # If False, plots will hide GT markers/labels even if JSON is present
PLOT_PREDICTIONS = True            # If False, hide predicted points/labels in plots
PRINT_ORIGINAL_COORDS = True       # If True, print predicted coords mapped back to original image pixels
PRINT_CROP_COORDS = True           # If True, also print predicted coords in crop pixel space
PRINT_TRANSFORM_DEBUG = False      # Print scale/pad/crop used for coordinate mapping
SAVE_ORIGINAL_OVERLAY = True       # If True, save overlay on original image in a separate folder
SAVE_CROPPED_OVERLAY = False       # If True, also save overlay on the cropped/resized display image
ORIGINAL_COORDS_MAP_MODE = "preproc"  # "input" -> use input image (w0,h0). "preproc" -> invert pad+crop
SHOW_CROP_BOX_ON_ORIGINAL = True   # Draw the projected crop rectangle on the original overlay
ORIGINAL_ALIGN_PRED_TO_GT_TRANSLATION = True  # If GT present, align predictions by mean translation (vis only)

##### EVALUATION CONTROL #####
# Final dataset-level evaluation will consider only visible keypoints (as per GT)
EVAL_PCK_THRESHOLDS_PX = [2, 5, 10, 15, 20]   # thresholds in pixels (IMG_SZ space)
EVAL_SAVE_JSON = True
EVAL_SAVE_TXT = True

##### VERTICES DETERMINATION #####
VERTICES_DETERMINE = False         # If True, override coord head using heatmap peaks
# Peak detection params (operate on heatmap resolution)
PEAK_THRESHOLD_REL = 0.15          # Keep peaks with value >= this fraction of per-vertex max
MAX_PEAKS_PER_VERTEX = 4           # Upper bound on number of candidates per vertex
NMS_SUPPRESS_RADIUS = 3            # Suppress neighborhood (pixels on heatmap) after a peak is found
CENTROID_WINDOW_RADIUS = 3         # Window radius (heatmap px) to compute intensity-weighted centroid
# How to determine the center of a local peak: "quadfit" or "connected"
PEAK_CENTER_MODE = "quadfit"
SUBPIXEL_PLATEAU_EPS = 0.05        # If neighbors within this fraction of peak, disable subpixel shift
CENTROID_ENFORCE_DESCENT = True     # When True, connected centroid won't cross to higher-intensity neighbors
##### ASSIGNMENT PARAMETERS #####
MIN_VERTEX_SEPARATION_IMG_PX = 1  # Avoid assigning two vertices closer than this (in IMG_SZ pixels)
 # When centroiding around a local peak, only include pixels that are part of
 # the connected component around that peak with value >= this fraction of the
 # peak value. This avoids drifting toward nearby separate peaks.
CENTROID_MIN_FRAC_OF_PEAK = 0.55


# ─────────────────────────────── helpers ───────────────────────────────

def _list_pairs_from_preproc():
    img_dir = getattr(preproc, "IMG_DIR", None)
    json_dir = getattr(preproc, "JSON_DIR", None)
    if not img_dir or not json_dir:
        return []
    if not os.path.exists(img_dir) or not os.path.exists(json_dir):
        print(f"Error: IMG_DIR or JSON_DIR not found: {img_dir}, {json_dir}")
        return []
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.exists(os.path.join(json_dir, os.path.splitext(f)[0] + ".json"))
    ]
    files.sort()
    return [
        (os.path.join(img_dir, f), os.path.join(json_dir, os.path.splitext(f)[0] + ".json"))
        for f in files
    ]


def _list_pairs_custom(img_dir, json_dir):
    if not img_dir or not json_dir:
        return []
    if not os.path.exists(img_dir) or not os.path.exists(json_dir):
        print(f"Error: IMG_DIR or JSON_DIR not found: {img_dir}, {json_dir}")
        return []
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.exists(os.path.join(json_dir, os.path.splitext(f)[0] + ".json"))
    ]
    files.sort()
    return [
        (os.path.join(img_dir, f), os.path.join(json_dir, os.path.splitext(f)[0] + ".json"))
        for f in files
    ]


def _list_pairs_from_images_dir_only(img_dir):
    if not img_dir or not os.path.exists(img_dir):
        print(f"Error: IMAGES_DIR_TO_PREDICT not found: {img_dir}")
        return []
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort()
    return [ (os.path.join(img_dir, f), None) for f in files ]


def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _pred_norm_to_original_pixels(
    pred_norm: np.ndarray,
    image_path: str,
    crop_w: int,
    crop_h: int,
    pad_w: int,
    pad_h: int,
    crop_left: int,
    crop_top: int,
) -> np.ndarray:
    """Map normalized predictions back to input image pixels.

    If ORIGINAL_COORDS_MAP_MODE == "input":
        Treat pred_norm as normalized in [0,1] over the visible crop and directly scale to the
        input image size (w0, h0): x = pred_norm_x * (w0 - 1), y = pred_norm_y * (h0 - 1).

    Else ("preproc"):
        Invert the preprocessing pipeline: resize_with_pad → center crop.
    """
    try:
        with Image.open(image_path) as im:
            w0, h0 = im.size
    except Exception:
        # Fallback via TensorFlow if PIL fails
        try:
            img_bytes = tf.io.read_file(image_path)
            img_dec = tf.io.decode_image(img_bytes, channels=3)
            h0 = int(img_dec.shape[0])
            w0 = int(img_dec.shape[1])
        except Exception:
            # Unknown size; return NaNs with same shape
            nan_arr = np.full_like(pred_norm, np.nan, dtype=np.float32)
            return nan_arr

    mode = str(ORIGINAL_COORDS_MAP_MODE).lower()
    if mode == "input":
        x_orig = pred_norm[:, 0] * max(float(w0) - 1.0, 1.0)
        y_orig = pred_norm[:, 1] * max(float(h0) - 1.0, 1.0)
        return np.stack([x_orig, y_orig], axis=-1).astype(np.float32)
    else:
        # Resize-with-pad scale and integer padding (match TF's rounding behavior more closely)
        s = min(float(pad_w) / float(w0), float(pad_h) / float(h0))
        resized_w = int(round(float(w0) * s))
        resized_h = int(round(float(h0) * s))
        left_pad = int(round((float(pad_w) - float(resized_w)) * 0.5))
        top_pad = int(round((float(pad_h) - float(resized_h)) * 0.5))

        # From normalized [0,1] (crop) to crop pixels (align with GT convention: minus 0.5)
        x_crop = pred_norm[:, 0] * float(crop_w) - 0.5
        y_crop = pred_norm[:, 1] * float(crop_h) - 0.5

        # To padded canvas
        x_padded = x_crop + float(crop_left)
        y_padded = y_crop + float(crop_top)

        # To original image pixels (invert scaling and padding)
        x_orig = (x_padded - float(left_pad)) / max(s, 1e-8)
        y_orig = (y_padded - float(top_pad)) / max(s, 1e-8)

        return np.stack([x_orig, y_orig], axis=-1).astype(np.float32)


def _plot_original_overlay_with_legend(
    image_path: str,
    pred_orig_px: np.ndarray,
    gt_orig_px: np.ndarray | None,
    visibility_mask: np.ndarray | None,
    base_fname: str,
    output_dir: str,
    plot_predictions: bool,
    plot_ground_truth: bool,
    crop_box_orig: tuple[float, float, float, float] | None = None,
):
    """Draw predicted and/or GT vertices on the original image and save.

    - visibility_mask: if provided, only plot indices where mask is True
    - legend contains per-vertex predicted coordinates actually plotted
    """
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: could not open image for original overlay: {image_path}: {e}")
        return

    w0, h0 = im.size
    fig, ax = plt.subplots(figsize=(min(10, w0/120), min(10, h0/120)))
    ax.imshow(im)
    ax.axis('off')

    legend_lines = []

    num_v = pred_orig_px.shape[0]
    for i in range(num_v):
        is_vis = True if visibility_mask is None else bool(visibility_mask[i])

        if plot_predictions and is_vis:
            x_p, y_p = float(pred_orig_px[i, 0]), float(pred_orig_px[i, 1])
            ax.scatter(x_p, y_p, c='seagreen', marker='o', s=36, edgecolors='k', linewidths=0.5, zorder=3)
            ax.text(x_p + 3, y_p, str(i), color='white', fontsize=8, zorder=4,
                    bbox=dict(facecolor='seagreen', alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.25'))
            legend_lines.append(f"{i:2d}: ({x_p:.1f}, {y_p:.1f})")

        if plot_ground_truth and gt_orig_px is not None and is_vis:
            gx, gy = float(gt_orig_px[i, 0]), float(gt_orig_px[i, 1])
            ax.scatter(gx, gy, c='red', marker='x', s=40, lw=1.0, zorder=4)

    if crop_box_orig is not None and SHOW_CROP_BOX_ON_ORIGINAL:
        x0, y0, x1, y1 = crop_box_orig
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor='yellow', linewidth=1.5, zorder=2))

    if legend_lines:
        legend_text = "\n".join(legend_lines)
        ax.text(0.01, 0.99, legend_text, transform=ax.transAxes, fontsize=8, color='white',
                va='top', ha='left', zorder=5,
                bbox=dict(facecolor='black', alpha=0.55, pad=4, edgecolor='none'))

    _ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"{base_fname}_orig_overlay.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved original overlay: {out_path}")


def _resolve_original_image_path(current_img_path: str) -> str:
    """Best-effort resolution of the original, full-size image path.

    Strategy:
    1) If '/Cropped/' appears in the current path, try replacing it with '/'.
    2) Try parent directory of Cropped ('.../Predictions') using same basename.
    3) Fallback to the current path.
    """
    stem, ext = os.path.splitext(os.path.basename(current_img_path))
    candidate_exts = [ext.lower(), ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def find_in_dir(d: str) -> str | None:
        for e in candidate_exts:
            p = os.path.join(d, stem + e)
            if os.path.isfile(p):
                return p
        return None

    # 1) Replace '/Cropped/' with '/'
    if "/Cropped/" in current_img_path:
        p2 = current_img_path.replace("/Cropped/", "/")
        if os.path.isfile(p2):
            return p2

    # 2) Parent of Cropped (Predictions)
    d = os.path.dirname(current_img_path)
    if os.path.basename(d).lower() == "cropped":
        parent = os.path.dirname(d)
        p3 = find_in_dir(parent)
        if p3:
            return p3

    # Fallback
    return current_img_path


# ────────────────────────  peak/centroid determination  ───────────────────────
def _nms_peak_candidates(
    heatmap_2d: np.ndarray,
    threshold_rel: float,
    max_peaks: int,
    suppress_radius: int,
    centroid_radius: int,
    centroid_min_frac_of_peak: float,
    center_mode: str,
    enforce_descent: bool,
):
    """Return list of peak candidates with intensity-weighted centroids.

    Each candidate is a dict: {"x": float, "y": float, "score": float} in heatmap pixel coords.
    """
    if heatmap_2d.size == 0:
        return []
    hm = np.asarray(heatmap_2d, dtype=np.float32)
    h, w = hm.shape
    if h == 0 or w == 0:
        return []
    work = hm.copy()
    vmax_global = float(hm.max())
    if vmax_global <= 0.0:
        return []
    threshold_abs = float(threshold_rel) * vmax_global

    peaks = []
    for _ in range(int(max(1, int(max_peaks)))):
        flat_idx = int(np.argmax(work))
        peak_val = float(work.flat[flat_idx])
        if peak_val < threshold_abs or peak_val <= 1e-12:
            break
        py, px = np.unravel_index(flat_idx, (h, w))
        # Choose center estimation method
        mode = (center_mode or "").lower()
        if mode == "quadfit":
            cx, cy = _subpixel_quadratic_from_3x3(hm, px, py)
        else:
            # Connected-component centroid within a local window, restricted to
            # values >= centroid_min_frac_of_peak * peak_val
            cx, cy = _centroid_connected_around_peak(
                hm, px, py, centroid_radius, centroid_min_frac_of_peak, enforce_descent
            )

        peaks.append({"x": float(cx), "y": float(cy), "score": peak_val})

        # Suppress a neighborhood around the peak in the working copy
        sx0 = max(0, px - suppress_radius)
        sx1 = min(w, px + suppress_radius + 1)
        sy0 = max(0, py - suppress_radius)
        sy1 = min(h, py + suppress_radius + 1)
        work[sy0:sy1, sx0:sx1] = 0.0

    return peaks


def _centroid_connected_around_peak(
    hm: np.ndarray,
    px: int,
    py: int,
    centroid_radius: int,
    min_frac_of_peak: float,
    enforce_descent: bool,
) -> tuple[float, float]:
    h, w = hm.shape
    if not (0 <= px < w and 0 <= py < h):
        return float(px), float(py)
    peak_val = float(hm[py, px])
    if peak_val <= 0.0:
        return float(px), float(py)
    thresh = float(min_frac_of_peak) * peak_val

    x0 = max(0, px - centroid_radius)
    x1 = min(w, px + centroid_radius + 1)
    y0 = max(0, py - centroid_radius)
    y1 = min(h, py + centroid_radius + 1)

    win_w = x1 - x0
    win_h = y1 - y0
    start_lx = px - x0
    start_ly = py - y0

    visited = np.zeros((win_h, win_w), dtype=np.uint8)
    q = deque()
    q.append((start_ly, start_lx))
    visited[start_ly, start_lx] = 1

    sumw = 0.0
    sumx = 0.0
    sumy = 0.0

    while q:
        ly, lx = q.popleft()
        gy = y0 + ly
        gx = x0 + lx
        v = float(hm[gy, gx])
        if v < thresh:
            continue
        sumw += v
        sumx += gx * v
        sumy += gy * v

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nly = ly + dy
                nlx = lx + dx
                if 0 <= nly < win_h and 0 <= nlx < win_w and not visited[nly, nlx]:
                    nv = float(hm[y0 + nly, x0 + nlx])
                    if nv >= thresh and (not enforce_descent or nv <= v + 1e-9 or (nly == start_ly and nlx == start_lx)):
                        visited[nly, nlx] = 1
                        q.append((nly, nlx))

    if sumw > 1e-12:
        return float(sumx / sumw), float(sumy / sumw)
    return float(px), float(py)


def _subpixel_quadratic_from_3x3(hm: np.ndarray, px: int, py: int) -> tuple[float, float]:
    """Return subpixel peak using 1D quadratic fit in x and y over a 3x3 patch.

    Falls back to the integer peak if near borders or if denominators are unstable.
    """
    h, w = hm.shape
    cx = float(px)
    cy = float(py)
    if not (1 <= px < w - 1 and 1 <= py < h - 1):
        return cx, cy
    f00 = float(hm[py, px])
    fm10 = float(hm[py, px - 1])
    fp10 = float(hm[py, px + 1])
    f0m1 = float(hm[py - 1, px])
    f0p1 = float(hm[py + 1, px])

    # If local neighborhood forms a near-plateau, avoid subpixel drift
    peak_eps = float(SUBPIXEL_PLATEAU_EPS) * max(1e-6, f00)
    if (abs(fm10 - f00) < peak_eps and abs(fp10 - f00) < peak_eps) or \
       (abs(f0m1 - f00) < peak_eps and abs(f0p1 - f00) < peak_eps):
        return cx, cy

    denom_x = (fm10 - 2.0 * f00 + fp10)
    denom_y = (f0m1 - 2.0 * f00 + f0p1)
    dx = 0.0
    dy = 0.0
    if abs(denom_x) > 1e-12:
        dx = 0.5 * (fm10 - fp10) / denom_x
    if abs(denom_y) > 1e-12:
        dy = 0.5 * (f0m1 - f0p1) / denom_y
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return cx + dx, cy + dy


def _determine_vertices_from_heatmaps(
    heatmaps_bhwc: np.ndarray,
    softargmax_norm_xy: np.ndarray,
    img_sz: int,
    heat_sz: int,
    min_separation_img_px: float,
    threshold_rel: float,
    max_peaks_per_vertex: int,
    suppress_radius: int,
    centroid_radius: int,
    centroid_min_frac_of_peak: float,
    center_mode: str,
):
    """Compute per-vertex coordinates based purely on heatmap peaks with conflict avoidance.

    Returns: np.ndarray shape (NUM_V, 2) normalized to [0,1] like the coord head.
    """
    assert heatmaps_bhwc.ndim == 4 and heatmaps_bhwc.shape[0] == 1, "Expect shape (1, H, W, N)"
    _, H, W, N = heatmaps_bhwc.shape

    # Safety
    if N <= 0:
        return softargmax_norm_xy.copy()

    # Pre-scale factors (derive denom from heatmap spatial dims for robustness)
    denom = max(float(H - 1), 1.0)
    img_scale = float(img_sz)

    candidates_per_vertex = []  # list[list[dict]]
    for vi in range(N):
        hm = np.asarray(heatmaps_bhwc[0, :, :, vi], dtype=np.float32)
        peaks = _nms_peak_candidates(
            hm,
            threshold_rel=threshold_rel,
            max_peaks=max_peaks_per_vertex,
            suppress_radius=suppress_radius,
            centroid_radius=centroid_radius,
            centroid_min_frac_of_peak=centroid_min_frac_of_peak,
            center_mode=center_mode,
        )

        # If no peaks, fallback to softargmax prediction as a weak candidate
        if not peaks:
            fx = float(softargmax_norm_xy[vi, 0]) * denom
            fy = float(softargmax_norm_xy[vi, 1]) * denom
            peaks = [{"x": fx, "y": fy, "score": 0.0}]

        # Augment with normalized and image-px coords
        augmented = []
        for p in peaks:
            x_hm = float(p["x"])  # in heatmap pixel coords
            y_hm = float(p["y"])
            x_norm = x_hm / denom
            y_norm = y_hm / denom
            augmented.append({
                "x_hm": x_hm,
                "y_hm": y_hm,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "x_img": x_norm * img_scale,
                "y_img": y_norm * img_scale,
                "score": float(p["score"]),
            })
        # Sort by score descending (stronger peaks first)
        augmented.sort(key=lambda d: d["score"], reverse=True)
        candidates_per_vertex.append(augmented)

    # Assignment
    assigned_norm = [None] * N  # type: ignore
    assigned_img_points = []     # list of (x_img, y_img)

    # First, assign vertices with exactly one candidate (single-possibility)
    single_indices = [i for i in range(N) if len(candidates_per_vertex[i]) == 1]
    for i in single_indices:
        c = candidates_per_vertex[i][0]
        assigned_norm[i] = np.array([c["x_norm"], c["y_norm"],], dtype=np.float32)
        assigned_img_points.append((c["x_img"], c["y_img"]))

    # Then, assign the remaining using conflict avoidance
    multi_indices = [i for i in range(N) if len(candidates_per_vertex[i]) > 1]
    # Process in order of best candidate score (desc)
    multi_indices.sort(key=lambda i: candidates_per_vertex[i][0]["score"], reverse=True)

    def far_enough(xy_img, others, min_dist):
        if not others:
            return True
        x, y = xy_img
        for (ox, oy) in others:
            dx = x - ox
            dy = y - oy
            if (dx * dx + dy * dy) < (min_dist * min_dist):
                return False
        return True

    for i in multi_indices:
        chosen = None
        for cand in candidates_per_vertex[i]:
            if far_enough((cand["x_img"], cand["y_img"]), assigned_img_points, min_separation_img_px):
                chosen = cand
                break
        if chosen is None:
            # All too close; pick the strongest candidate
            chosen = candidates_per_vertex[i][0]
        assigned_norm[i] = np.array([chosen["x_norm"], chosen["y_norm"]], dtype=np.float32)
        assigned_img_points.append((chosen["x_img"], chosen["y_img"]))

    # Finally, for any vertex that somehow remained unassigned, fallback to softargmax
    for i in range(N):
        if assigned_norm[i] is None:
            assigned_norm[i] = np.array([softargmax_norm_xy[i, 0], softargmax_norm_xy[i, 1]], dtype=np.float32)

    return np.stack(assigned_norm, axis=0)


def _plot_overlay_and_heatmaps(
    img_for_plot,
    gt_norm,
    pred_norm,
    heatmaps_pred,
    sample_idx,
    base_fname,
    output_dir,
    img_sz,
    num_vertices,
):
    _ensure_dir(output_dir)

    # Overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(img_for_plot, 0, 1))
    ax.axis('off')

    legend_handles = {}
    invisible_indices = []

    for i in range(num_vertices):
        is_visible_gt = False
        if gt_norm is not None:
            is_visible_gt = (gt_norm[i, 0] >= 0) and (gt_norm[i, 1] >= 0)

        pred_px, pred_py = pred_norm[i, 0] * img_sz, pred_norm[i, 1] * img_sz

        if PLOT_PREDICTIONS and (gt_norm is None or is_visible_gt):
            h_pred = ax.scatter(pred_px, pred_py, c='seagreen', marker='o', s=60, edgecolors='k', linewidths=0.5, label='Prediction', zorder=4)
            if 'Prediction' not in legend_handles:
                legend_handles['Prediction'] = h_pred
            ax.text(pred_px + 3, pred_py, str(i), color='white', fontsize=9, zorder=5,
                    bbox=dict(facecolor='seagreen', alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.25'))

        if is_visible_gt and PLOT_GROUND_TRUTH:
            gt_px, gt_py = gt_norm[i, 0] * img_sz, gt_norm[i, 1] * img_sz
            h_gt = ax.scatter(gt_px, gt_py, c='red', marker='x', s=60, lw=1.5, label='Ground Truth', zorder=3)
            if 'Ground Truth' not in legend_handles:
                legend_handles['Ground Truth'] = h_gt
        elif PLOT_GROUND_TRUTH:
            invisible_indices.append(str(i))

    if PLOT_GROUND_TRUTH and invisible_indices:
        inv_text = "Invisible: " + ", ".join(invisible_indices)
        ax.text(0.02, 0.98, inv_text, transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6, pad=3, edgecolor='none'))

    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()), loc='best', fontsize='small')

    fig.tight_layout()
    _ensure_dir(output_dir)
    plt.savefig(os.path.join(output_dir, f"{base_fname}_overlay.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Per-vertex heatmaps
    num_cols = 4
    num_rows = int(np.ceil(num_vertices / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.0, num_rows * 3.0))
    axes = axes.flatten()
    for i in range(num_vertices):
        ax = axes[i]
        hm_resized = tf.image.resize(heatmaps_pred[0, :, :, i:i+1], (img_sz, img_sz)).numpy().squeeze()
        ax.imshow(np.clip(img_for_plot, 0, 1), alpha=0.4)
        ax.imshow(hm_resized, cmap='hot', alpha=0.6)
        title_info = f"Vtx {i}"
        if PLOT_GROUND_TRUTH and gt_norm is not None and not ((gt_norm[i, 0] >= 0) and (gt_norm[i, 1] >= 0)):
            title_info += " (inv)"
        ax.set_title(title_info)
        ax.axis('off')
        # predicted point
        if PLOT_PREDICTIONS and (gt_norm is None or ((gt_norm[i, 0] >= 0) and (gt_norm[i, 1] >= 0))):
            px, py = pred_norm[i, 0] * img_sz, pred_norm[i, 1] * img_sz
            ax.scatter(px, py, c='cyan', marker='o', s=36, edgecolors='k', linewidths=0.4, zorder=4)
        # ground truth if visible
        if PLOT_GROUND_TRUTH and gt_norm is not None and (gt_norm[i, 0] >= 0) and (gt_norm[i, 1] >= 0):
            gx, gy = gt_norm[i, 0] * img_sz, gt_norm[i, 1] * img_sz
            ax.scatter(gx, gy, c='red', marker='x', s=40, lw=1.0, zorder=4)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Sample {sample_idx}: {base_fname} - Heatmaps", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{base_fname}_heatmaps.png"), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ────────────────────────────  main inference  ───────────────────────────
def main():
    configure_gpu()

    IMG_SZ = int(getattr(preproc, "IMG_SZ", 128))
    HEAT_SZ = int(getattr(preproc, "HEAT_SZ", 64))
    NUM_V = int(getattr(preproc, "NUM_V", 8))

    # Optional user overrides
    if FORCE_IMG_SZ is not None:
        IMG_SZ = int(FORCE_IMG_SZ)
    if FORCE_HEAT_SZ is not None:
        HEAT_SZ = int(FORCE_HEAT_SZ)
    if FORCE_NUM_V is not None:
        NUM_V = int(FORCE_NUM_V)

    # Choose model path: prefer best checkpoint, else final saved model
    if PRIORITIZE_BEST_CHECKPOINT:
        model_path_candidates = [MODEL_CKPT_PATH, MODEL_FINAL_PATH]
    else:
        model_path_candidates = [MODEL_FINAL_PATH, MODEL_CKPT_PATH]
    model_path = None
    for p in model_path_candidates:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        print("❌ No model file found. Expected one of:")
        for p in model_path_candidates:
            print(f" - {p}")
        return

    custom_objects = {"SoftArgMax": SoftArgMax, "focal_bce": focal_bce, "masked_l1": masked_l1}

    model_loaded_successfully = False
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=True)
        print(f"✅ Model loaded (compiled) from: {model_path}\\n")
        model_loaded_successfully = True
    except Exception as e_compile_true:
        print(f"Info: Could not load model with compile=True (Error: {e_compile_true}).")
        print("Attempting to load with compile=False.")
        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"✅ Model loaded (compile=False) from: {model_path}\\n")
            model_loaded_successfully = True
        except Exception as e_compile_false:
            print(f"❌ Error: Failed to load model with compile=False. Error: {e_compile_false}")
            return

    if not model_loaded_successfully:
        print("❌ Model could not be loaded.")
        return

    # Detect NUM_V from model outputs (coords head is rank-2, shape [None, NUM_V*2])
    if FORCE_NUM_V is None:
        try:
            coords_out = None
            for o in model.outputs:
                # coords head is rank-2, heatmaps is rank-4
                if len(o.shape) == 2:
                    coords_out = o
                    break
            if coords_out is not None and coords_out.shape[-1] is not None:
                num_coords = int(coords_out.shape[-1])
                if num_coords % 2 == 0:
                    NUM_V_MODEL = num_coords // 2
                    if NUM_V_MODEL != NUM_V:
                        print(f"Info: Overriding NUM_V from preprocessing ({NUM_V}) to model-detected {NUM_V_MODEL}.")
                        NUM_V = NUM_V_MODEL
            else:
                print("Warning: Could not infer NUM_V from model outputs; using preprocessing value.")
        except Exception as e:
            print(f"Warning: Failed to inspect model outputs for NUM_V: {e}")

    if IMAGES_DIR_TO_PREDICT:
        pairs = _list_pairs_from_images_dir_only(IMAGES_DIR_TO_PREDICT)
    else:
        pairs = _list_pairs_custom(CUSTOM_IMG_DIR, CUSTOM_JSON_DIR)
    if not pairs:
        print("❌ No images found. Set IMAGES_DIR_TO_PREDICT for images-only, or set CUSTOM_IMG_DIR and CUSTOM_JSON_DIR.")
        return

    if isinstance(LIMIT_SAMPLES, int) and LIMIT_SAMPLES > 0:
        pairs = pairs[:LIMIT_SAMPLES]

    _ensure_dir(OUTPUT_VIS_DIR)
    overlay_dir = os.path.join(OUTPUT_VIS_DIR, "overlays")
    heatmaps_dir = os.path.join(OUTPUT_VIS_DIR, "heatmaps")
    original_overlay_dir = os.path.join(OUTPUT_VIS_DIR, "original_overlays")
    _ensure_dir(overlay_dir)
    _ensure_dir(heatmaps_dir)
    if SAVE_ORIGINAL_OVERLAY:
        _ensure_dir(original_overlay_dir)

    all_maes_visible = []
    # Dataset-level evaluation accumulators (visible keypoints only, in IMG_SZ pixels)
    eval_abs_dx = []     # list of float
    eval_abs_dy = []     # list of float
    eval_euclidean = []  # list of float
    total_visible_points = 0
    num_samples_with_gt = 0
    num_samples_processed = 0
    print(f"Running inference on {len(pairs)} samples. Outputs → {OUTPUT_VIS_DIR}")

    for idx, (img_path, jpath) in enumerate(pairs, start=1):
        base_fname = os.path.splitext(os.path.basename(img_path))[0]
        crop_box_orig = None  # default; may be computed later

        try:
            x_ready, img_disp = preproc.preprocess_image_for_model(img_path)
        except Exception as e:
            print(f"Warning: failed to preprocess {img_path}: {e}")
            continue

        try:
            pred_heatmaps, pred_coords_flat = model.predict(x_ready[None, ...], verbose=0)
        except Exception as e:
            print(f"Warning: prediction failed for {img_path}: {e}")
            continue

        pred_xy_norm = pred_coords_flat.reshape(NUM_V, 2)
        if VERTICES_DETERMINE:
            try:
                pred_xy_norm = _determine_vertices_from_heatmaps(
                    heatmaps_bhwc=pred_heatmaps,
                    softargmax_norm_xy=pred_xy_norm,
                    img_sz=IMG_SZ,
                    heat_sz=HEAT_SZ,
                    min_separation_img_px=float(MIN_VERTEX_SEPARATION_IMG_PX),
                    threshold_rel=float(PEAK_THRESHOLD_REL),
                    max_peaks_per_vertex=int(MAX_PEAKS_PER_VERTEX),
                    suppress_radius=int(NMS_SUPPRESS_RADIUS),
                    centroid_radius=int(CENTROID_WINDOW_RADIUS),
                    centroid_min_frac_of_peak=float(CENTROID_MIN_FRAC_OF_PEAK),
                    center_mode=str(PEAK_CENTER_MODE),
                    # enforce descent only when using connected centroid mode
                )
            except Exception as e:
                print(f"Warning: VERTICES_DETERMINE failed for {img_path}: {e}. Using model coords.")
        # Optionally print original-image pixel predictions
        if PRINT_ORIGINAL_COORDS:
            try:
                # Import offsets and sizes from preprocess module
                PADDED_HEIGHT = int(getattr(preproc, "PADDED_HEIGHT", 1080))
                PADDED_WIDTH = int(getattr(preproc, "PADDED_WIDTH", 1920))
                CROP_HEIGHT = int(getattr(preproc, "CROP_HEIGHT", IMG_SZ))
                CROP_WIDTH = int(getattr(preproc, "CROP_WIDTH", IMG_SZ))
                get_center_crop_offsets = getattr(preproc, "get_center_crop_offsets", None)
                if callable(get_center_crop_offsets):
                    crop_top, crop_left = get_center_crop_offsets()
                else:
                    crop_top, crop_left = 0, 0
                if PRINT_TRANSFORM_DEBUG:
                    print(f"Transform info: pad=({PADDED_WIDTH}x{PADDED_HEIGHT}), crop=({CROP_WIDTH}x{CROP_HEIGHT}) at (left={crop_left}, top={crop_top})")
                if PRINT_CROP_COORDS:
                    # Print prediction in crop pixel space (pre-normalization convention used in GT)
                    x_crop = pred_xy_norm[:, 0] * float(CROP_WIDTH) - 0.5
                    y_crop = pred_xy_norm[:, 1] * float(CROP_HEIGHT) - 0.5
                    print("Predicted vertices in CROP pixels (x,y):")
                    for vi in range(NUM_V):
                        print(f"  {vi:2d}: ({x_crop[vi]:8.2f}, {y_crop[vi]:8.2f})")
                orig_img_path = _resolve_original_image_path(img_path)
                pred_orig_px = _pred_norm_to_original_pixels(
                    pred_xy_norm,
                    image_path=orig_img_path,
                    crop_w=CROP_WIDTH,
                    crop_h=CROP_HEIGHT,
                    pad_w=PADDED_WIDTH,
                    pad_h=PADDED_HEIGHT,
                    crop_left=crop_left,
                    crop_top=crop_top,
                )
                # Compute and draw the projected crop rectangle on the original
                try:
                    with Image.open(orig_img_path) as _im_tmp:
                        w0_tmp, h0_tmp = _im_tmp.size
                    s = min(float(PADDED_WIDTH) / float(w0_tmp), float(PADDED_HEIGHT) / float(h0_tmp))
                    resized_w = int(round(float(w0_tmp) * s))
                    resized_h = int(round(float(h0_tmp) * s))
                    left_pad = int(round((float(PADDED_WIDTH) - float(resized_w)) * 0.5))
                    top_pad = int(round((float(PADDED_HEIGHT) - float(resized_h)) * 0.5))
                    # crop box in padded canvas coords
                    cx0 = float(crop_left)
                    cy0 = float(crop_top)
                    cx1 = cx0 + float(CROP_WIDTH)
                    cy1 = cy0 + float(CROP_HEIGHT)
                    # map to original: (padded - pad) / s
                    ox0 = (cx0 - float(left_pad)) / max(s, 1e-8)
                    oy0 = (cy0 - float(top_pad)) / max(s, 1e-8)
                    ox1 = (cx1 - float(left_pad)) / max(s, 1e-8)
                    oy1 = (cy1 - float(top_pad)) / max(s, 1e-8)
                    crop_box_orig = (ox0, oy0, ox1, oy1)
                except Exception:
                    crop_box_orig = None
                print("\\nPredicted vertices in ORIGINAL image pixels (x,y) [mode=", ORIGINAL_COORDS_MAP_MODE, "]:")
                for vi in range(NUM_V):
                    x_o, y_o = pred_orig_px[vi]
                    print(f"  {vi:2d}: ({x_o:8.2f}, {y_o:8.2f})")
            except Exception as e:
                print(f"Warning: failed to compute original pixel coords for {img_path}: {e}")
        if jpath is not None:
            try:
                gt_xy_norm = preproc.load_vertices(jpath)
            except Exception as e:
                print(f"Warning: failed to load GT vertices from {jpath}: {e}")
                gt_xy_norm = None
        else:
            gt_xy_norm = None

        if gt_xy_norm is not None:
            num_samples_with_gt += 1
            visible_mask = (gt_xy_norm[:, 0] >= 0) & (gt_xy_norm[:, 1] >= 0)
            gt_pixels = gt_xy_norm * float(IMG_SZ)
            pred_pixels = pred_xy_norm * float(IMG_SZ)

            errors_pixels = np.full_like(pred_pixels, -1.0, dtype=np.float32)
            if np.any(visible_mask):
                errors_pixels[visible_mask] = np.abs(pred_pixels[visible_mask] - gt_pixels[visible_mask])
                # Accumulate dataset-level metrics (visible only)
                vis_errs = errors_pixels[visible_mask]
                eval_abs_dx.extend(vis_errs[:, 0].tolist())
                eval_abs_dy.extend(vis_errs[:, 1].tolist())
                eval_euclidean.extend(np.sqrt(np.sum(np.square(vis_errs), axis=1)).tolist())
                total_visible_points += int(np.sum(visible_mask))

            print(f"\\nSample {idx:03d}: {base_fname}")
            print("-" * 70)
            print(f"{'Vtx':>3} | {'GT X':>7} {'GT Y':>7} | {'Pr X':>7} {'Pr Y':>7} | {'ErrX':>6} {'ErrY':>6} | Vis")
            print("-" * 70)

            current_sample_errors_visible_coords = []
            for i in range(NUM_V):
                is_visible = bool(visible_mask[i])
                p_px, p_py = pred_pixels[i]
                if is_visible:
                    g_px, g_py = gt_pixels[i]
                    e_px, e_py = errors_pixels[i]
                    print(f"{i:3d} | {g_px:7.1f} {g_py:7.1f} | {p_px:7.1f} {p_py:7.1f} | {e_px:6.1f} {e_py:6.1f} | V")
                    current_sample_errors_visible_coords.append(errors_pixels[i])
                else:
                    print(f"{i:3d} | {'  N/A':>7} {'  N/A':>7} | {p_px:7.1f} {p_py:7.1f} | {' N/A':>6} {' N/A':>6} | -")

            if current_sample_errors_visible_coords:
                sample_mae_visible = np.mean(current_sample_errors_visible_coords, axis=0)
                print("-" * 70)
                print(f" Sample MAE (visible, px): dx={sample_mae_visible[0]:.2f}, dy={sample_mae_visible[1]:.2f}\\n")
                all_maes_visible.append(sample_mae_visible)
            else:
                print("-" * 70)
                print(" Sample MAE (visible, px): No visible GT keypoints for this sample.\\n")
        else:
            print(f"\\nSample {idx:03d}: {base_fname}")
            print("(No JSON/GT provided. Skipping error table.)")

        # Save overlay on original image if enabled
        if SAVE_ORIGINAL_OVERLAY:
            try:
                PADDED_HEIGHT = int(getattr(preproc, "PADDED_HEIGHT", 1080))
                PADDED_WIDTH = int(getattr(preproc, "PADDED_WIDTH", 1920))
                CROP_HEIGHT = int(getattr(preproc, "CROP_HEIGHT", IMG_SZ))
                CROP_WIDTH = int(getattr(preproc, "CROP_WIDTH", IMG_SZ))
                get_center_crop_offsets = getattr(preproc, "get_center_crop_offsets", None)
                if callable(get_center_crop_offsets):
                    crop_top, crop_left = get_center_crop_offsets()
                else:
                    crop_top, crop_left = 0, 0
                pred_orig_px = _pred_norm_to_original_pixels(
                    pred_xy_norm,
                    image_path=img_path,
                    crop_w=CROP_WIDTH,
                    crop_h=CROP_HEIGHT,
                    pad_w=PADDED_WIDTH,
                    pad_h=PADDED_HEIGHT,
                    crop_left=crop_left,
                    crop_top=crop_top,
                )
                if gt_xy_norm is not None:
                    # Map GT back to original image too, mirroring load_vertices normalization
                    gx_crop = gt_xy_norm[:, 0] * float(CROP_WIDTH) - 0.5
                    gy_crop = gt_xy_norm[:, 1] * float(CROP_HEIGHT) - 0.5
                    gx_pad = gx_crop + float(crop_left)
                    gy_pad = gy_crop + float(crop_top)
                    # Invert resize_with_pad
                    s = min(float(PADDED_WIDTH) / 1.0, float(PADDED_HEIGHT) / 1.0)  # placeholder to define type
                    # Recompute exact scale as in helper
                    with Image.open(img_path) as _im_tmp:
                        w0_tmp, h0_tmp = _im_tmp.size
                    s = min(float(PADDED_WIDTH) / float(w0_tmp), float(PADDED_HEIGHT) / float(h0_tmp))
                    resized_w = int(round(float(w0_tmp) * s))
                    resized_h = int(round(float(h0_tmp) * s))
                    left_pad = int(round((float(PADDED_WIDTH) - float(resized_w)) * 0.5))
                    top_pad = int(round((float(PADDED_HEIGHT) - float(resized_h)) * 0.5))
                    gx_orig = (gx_pad - float(left_pad)) / max(s, 1e-8)
                    gy_orig = (gy_pad - float(top_pad)) / max(s, 1e-8)
                    gt_orig_px = np.stack([gx_orig, gy_orig], axis=-1).astype(np.float32)
                    # Optional visual-only alignment: correct any constant translation by matching GT mean
                    if ORIGINAL_ALIGN_PRED_TO_GT_TRANSLATION and pred_orig_px is not None:
                        vis_mask_np = (gt_xy_norm[:, 0] >= 0) & (gt_xy_norm[:, 1] >= 0)
                        if np.any(vis_mask_np):
                            delta = np.mean(gt_orig_px[vis_mask_np] - pred_orig_px[vis_mask_np], axis=0)
                            pred_orig_px = pred_orig_px + delta
                else:
                    gt_orig_px = None
                _plot_original_overlay_with_legend(
                    image_path=_resolve_original_image_path(img_path),
                    pred_orig_px=pred_orig_px,
                    gt_orig_px=gt_orig_px,
                    visibility_mask=visible_mask if gt_xy_norm is not None else None,
                    base_fname=base_fname,
                    output_dir=original_overlay_dir,
                    plot_predictions=bool(PLOT_PREDICTIONS),
                    plot_ground_truth=bool(PLOT_GROUND_TRUTH and gt_xy_norm is not None),
                    crop_box_orig=crop_box_orig,
                )
            except Exception as e:
                print(f"Warning: failed to save original overlay for {img_path}: {e}")

        # Visualize first N samples by setting VISUALIZE_FIRST_N
        if SAVE_CROPPED_OVERLAY and isinstance(VISUALIZE_FIRST_N, int) and VISUALIZE_FIRST_N > 0 and idx <= VISUALIZE_FIRST_N:
            _plot_overlay_and_heatmaps(
                img_for_plot=img_disp,
                gt_norm=gt_xy_norm,
                pred_norm=pred_xy_norm,
                heatmaps_pred=pred_heatmaps,
                sample_idx=idx,
                base_fname=base_fname,
                output_dir=os.path.join(OUTPUT_VIS_DIR, base_fname),
                img_sz=IMG_SZ,
                num_vertices=NUM_V,
            )

        # Count processed samples (those that reached this point without early continue)
        num_samples_processed += 1

    if all_maes_visible:
        overall_mae_visible = np.mean(all_maes_visible, axis=0)
        print("=" * 70)
        print(f"Overall MAE for VISIBLE keypoints (pixels): dx = {overall_mae_visible[0]:.2f}, dy = {overall_mae_visible[1]:.2f}")
        print("=" * 70)
    else:
        print("=" * 70)
        print("No visible keypoints found across processed samples to compute overall MAE.")
        print("=" * 70)

    # ─────────────── Final dataset-level evaluation (visible keypoints only) ───────────────
    if len(eval_euclidean) > 0:
        abs_dx_arr = np.asarray(eval_abs_dx, dtype=np.float32)
        abs_dy_arr = np.asarray(eval_abs_dy, dtype=np.float32)
        eucl_arr = np.asarray(eval_euclidean, dtype=np.float32)

        mae_dx = float(np.mean(abs_dx_arr))
        mae_dy = float(np.mean(abs_dy_arr))
        rmse_dx = float(np.sqrt(np.mean(np.square(abs_dx_arr))))
        rmse_dy = float(np.sqrt(np.mean(np.square(abs_dy_arr))))

        eucl_mean = float(np.mean(eucl_arr))
        eucl_median = float(np.median(eucl_arr))
        eucl_p95 = float(np.percentile(eucl_arr, 95))
        eucl_rmse = float(np.sqrt(np.mean(np.square(eucl_arr))))

        # PCK at given pixel thresholds (percentage of visible keypoints within threshold)
        pck = {}
        for thr in EVAL_PCK_THRESHOLDS_PX:
            thr_f = float(thr)
            p = float(np.mean(eucl_arr <= thr_f)) * 100.0
            pck[str(thr)] = p

        print("\\n" + "#" * 70)
        print("Final dataset evaluation (visible keypoints; pixel units in IMG_SZ space)")
        print(f"Processed samples: {num_samples_processed}")
        print(f"Samples with GT:   {num_samples_with_gt}")
        print(f"Visible points:    {total_visible_points}")
        print("-" * 70)
        print(f"MAE (dx, dy):      ({mae_dx:.2f}, {mae_dy:.2f})")
        print(f"RMSE (dx, dy):     ({rmse_dx:.2f}, {rmse_dy:.2f})")
        print(f"Euclidean mean:    {eucl_mean:.2f} px")
        print(f"Euclidean median:  {eucl_median:.2f} px")
        print(f"Euclidean P95:     {eucl_p95:.2f} px")
        print(f"Euclidean RMSE:    {eucl_rmse:.2f} px")
        print("PCK (percent within threshold px):")
        for thr in EVAL_PCK_THRESHOLDS_PX:
            print(f"  ≤ {thr:>3}px : {pck[str(thr)]:6.2f}%")
        print("#" * 70 + "\\n")

        # Save summary
        summary = {
            "img_size_px": int(IMG_SZ),
            "processed_samples": int(num_samples_processed),
            "samples_with_gt": int(num_samples_with_gt),
            "visible_keypoints_count": int(total_visible_points),
            "mae_dx": mae_dx,
            "mae_dy": mae_dy,
            "rmse_dx": rmse_dx,
            "rmse_dy": rmse_dy,
            "euclidean_mean": eucl_mean,
            "euclidean_median": eucl_median,
            "euclidean_p95": eucl_p95,
            "euclidean_rmse": eucl_rmse,
            "pck_thresholds_px": list(EVAL_PCK_THRESHOLDS_PX),
            "pck_percent": pck,
        }
        try:
            if EVAL_SAVE_JSON:
                json_path = os.path.join(OUTPUT_VIS_DIR, "evaluation_summary.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved evaluation JSON: {json_path}")
            if EVAL_SAVE_TXT:
                txt_path = os.path.join(OUTPUT_VIS_DIR, "evaluation_summary.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("Final dataset evaluation (visible keypoints; pixel units in IMG_SZ space)\\n")
                    f.write(f"Processed samples: {num_samples_processed}\\n")
                    f.write(f"Samples with GT:   {num_samples_with_gt}\\n")
                    f.write(f"Visible points:    {total_visible_points}\\n")
                    f.write("-" * 70 + "\\n")
                    f.write(f"MAE (dx, dy):      ({mae_dx:.2f}, {mae_dy:.2f})\\n")
                    f.write(f"RMSE (dx, dy):     ({rmse_dx:.2f}, {rmse_dy:.2f})\\n")
                    f.write(f"Euclidean mean:    {eucl_mean:.2f} px\\n")
                    f.write(f"Euclidean median:  {eucl_median:.2f} px\\n")
                    f.write(f"Euclidean P95:     {eucl_p95:.2f} px\\n")
                    f.write(f"Euclidean RMSE:    {eucl_rmse:.2f} px\\n")
                    f.write("PCK (percent within threshold px):\\n")
                    for thr in EVAL_PCK_THRESHOLDS_PX:
                        f.write(f"  ≤ {thr:>3}px : {pck[str(thr)]:6.2f}%\\n")
                print(f"Saved evaluation TXT: {txt_path}")
        except Exception as e:
            print(f"Warning: failed to save evaluation summary files: {e}")
    else:
        print("No visible keypoints across dataset; final evaluation metrics not computed.")


if __name__ == "__main__":
    main()
