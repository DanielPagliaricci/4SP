#!/usr/bin/env python3
# Preprocess only: build X/Y numpy datasets for heatmap + coords labels

import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ─────────────────────────  paths / params  ─────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(BASE_DIR)
IMG_DIR   = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Rendered_Images")
JSON_DIR  = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "JSON_Data_REF")


MESH_INFO_PATH = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Object_Information", "mesh_information_REF_TANGO.json")

LOG_DIR   = os.path.join(BASE_DIR, "plots")
OUTPUT_DIR = BASE_DIR

IMG_SZ, HEAT_SZ = 128, 64
SIGMA_PX  = 2.0

# Padded canvas size (after resize_with_pad)
PADDED_HEIGHT: int = 1080
PADDED_WIDTH: int = 1920

# Crop hyperparameters (size only). Position is auto-centered.
CROP_HEIGHT: int = 1000
CROP_WIDTH: int = 1000

def get_center_crop_offsets():
    top = max(0, (PADDED_HEIGHT - CROP_HEIGHT) // 2)
    left = max(0, (PADDED_WIDTH - CROP_WIDTH) // 2)
    return top, left

def load_num_vertices(info_path: str, default_num_vertices: int = 8) -> int:
    try:
        with open(info_path, "r") as f:
            data = json.load(f)
        num_vertices = int(data.get("total_vertices", default_num_vertices))
        if num_vertices <= 0:
            raise ValueError("total_vertices must be positive")
        return num_vertices
    except Exception as e:
        print(f"Warning: failed to load total_vertices from {info_path}. Using default {default_num_vertices}. Error: {e}")
        return default_num_vertices

NUM_V = load_num_vertices(MESH_INFO_PATH, default_num_vertices=8)

# ═══════════════════════════  data helpers  ════════════════════════════
def vertices_to_heatmaps(v_norm): # v_norm is (NUM_V, 2)
    hm = np.zeros((HEAT_SZ, HEAT_SZ, NUM_V), np.float32)
    xv, yv = np.meshgrid(np.arange(HEAT_SZ), np.arange(HEAT_SZ))
    for i, (x_n, y_n) in enumerate(v_norm.reshape(NUM_V, 2)):
        # Treat as invisible if not inside normalized crop [0,1]
        if x_n < 0 or y_n < 0 or x_n > 1 or y_n > 1:
            continue
        # Denormalize to heatmap coordinates
        cx, cy = x_n * (HEAT_SZ -1) , y_n * (HEAT_SZ -1) # Scale to [0, HEAT_SZ-1]
        hm[..., i] = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * SIGMA_PX**2))
    return hm

def load_vertices(path):
    with open(path) as f:
        d = json.load(f)
    top, left = get_center_crop_offsets()
    crop_w, crop_h = CROP_WIDTH, CROP_HEIGHT
    v = np.full((NUM_V, 2), -1.0, np.float32) # Use -1.0 for float type consistency
    for p in d.get("visible_vertices", []):
        i, (px, py) = p["index"], p["pixel_coordinates"]
        if 0 <= i < NUM_V:
            # Convert to local crop coordinates
            lx = px - left
            ly = py - top
            # Only keep as visible if inside the crop box
            if 0 <= lx < crop_w and 0 <= ly < crop_h:
                v[i] = [(lx + 0.5) / crop_w, (ly + 0.5) / crop_h]  # normalize to [0,1]
            else:
                # Stays -1.0 meaning non-visible in the cropped view
                pass
    return v

def list_image_json_pairs():
    files = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.exists(os.path.join(JSON_DIR, os.path.splitext(f)[0] + ".json"))
    ]
    files.sort()
    ipaths = [os.path.join(IMG_DIR, f) for f in files]
    jpaths = [os.path.join(JSON_DIR, os.path.splitext(f)[0] + ".json") for f in files]
    return list(zip(ipaths, jpaths))

def preprocess_image_for_model(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_bytes, channels=3, dtype=tf.uint8)
    img.set_shape([None, None, 3])
    img_float = tf.image.convert_image_dtype(img, tf.float32)
    img_padded = tf.image.resize_with_pad(img_float, PADDED_HEIGHT, PADDED_WIDTH)
    top, left = get_center_crop_offsets()
    img_cropped = tf.image.crop_to_bounding_box(img_padded, top, left, CROP_HEIGHT, CROP_WIDTH)
    img_resized = tf.image.resize(img_cropped, (IMG_SZ, IMG_SZ))
    img_scaled = img_resized * 255.0
    img_ready = preprocess_input(img_scaled)
    return img_ready.numpy(), img_resized.numpy()  # (model_input, display_float_[0,1])

def plot_image_with_vertices(img_float01, v_norm, out_path, title=None):
    vis_mask = (
        (v_norm[:, 0] >= 0) & (v_norm[:, 0] <= 1) &
        (v_norm[:, 1] >= 0) & (v_norm[:, 1] <= 1)
    )
    xs = v_norm[vis_mask, 0] * (IMG_SZ - 1)
    ys = v_norm[vis_mask, 1] * (IMG_SZ - 1)
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(np.clip(img_float01, 0, 1))
    if xs.size > 0:
        plt.scatter(xs, ys, c='lime', s=24, marker='o', label='visible')
        # annotate indices
        vis_indices = np.nonzero(vis_mask)[0]
        for (x, y, i) in zip(xs, ys, vis_indices):
            plt.text(x + 2, y - 2, str(i), color='lime', fontsize=8)
    # list non-visible indices
    non_vis_indices = np.where(~vis_mask)[0]
    if non_vis_indices.size > 0:
        txt = "non-visible: " + ", ".join(map(str, non_vis_indices.tolist()))
        plt.gca().text(2, 12, txt, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.axis('off')
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def write_npy_datasets():
    pairs = list_image_json_pairs()
    if not pairs:
        print("No image/json pairs found. Check IMG_DIR and JSON_DIR.")
        return
    total = len(pairs)
    print(f"Found {total} image/json pairs. Starting preprocessing...")
    X_list, H_list, C_list = [], [], []
    iterator = tqdm(pairs, desc="Preprocessing", unit="pair") if tqdm is not None else enumerate(pairs, 1)
    for item in iterator:
        if tqdm is not None:
            ip, jp = item
        else:
            idx, (ip, jp) = item
        x_ready, img_disp = preprocess_image_for_model(ip)
        vv = load_vertices(jp)
        heat = vertices_to_heatmaps(vv)
        coords = vv.flatten()
        X_list.append(x_ready.astype(np.float32))
        H_list.append(heat.astype(np.float32))
        C_list.append(coords.astype(np.float32))
        base = os.path.splitext(os.path.basename(ip))[0]
        plot_path = os.path.join(LOG_DIR, f"{base}_vertices.png")
        plot_image_with_vertices(img_disp, vv, plot_path)
        if tqdm is None:
            # Lightweight progress output without external deps
            if total:
                percent = int(idx * 100 / total)
                if idx == 1 or idx == total or idx % max(1, total // 50) == 0:
                    print(f"Preprocessing: {idx}/{total} ({percent}%)", end="\r")
    if tqdm is None:
        print()  # newline after carriage-return updates
    X_data = np.stack(X_list, axis=0)
    Y_heat = np.stack(H_list, axis=0)
    Y_coords = np.stack(C_list, axis=0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "Y_data.npz")
    np.save(x_path, X_data)
    # Save Y as NPZ (separate arrays, no pickle; avoids 4 GiB pickle protocol limit)
    np.savez_compressed(y_path, heatmaps=Y_heat, coords=Y_coords)
    print(f"Saved X_data: {X_data.shape} → {x_path}")
    print(f"Saved Y_data (NPZ): heatmaps {Y_heat.shape}, coords {Y_coords.shape} → {y_path}")

def visualize_precheck_grid(num_samples=4):
    pairs = list_image_json_pairs()
    if not pairs:
        print("No image/json pairs found. Skipping precheck grid.")
        return
    n = min(num_samples, len(pairs))
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(n):
        ip, jp = pairs[i]
        x_ready, img_disp = preprocess_image_for_model(ip)
        vv = load_vertices(jp)
        heat = vertices_to_heatmaps(vv)
        heat_vis = np.max(heat, axis=-1)
        heat_vis_resized = tf.image.resize(heat_vis[..., None], (IMG_SZ, IMG_SZ), method="bilinear").numpy().squeeze()
        base = os.path.splitext(os.path.basename(ip))[0]
        axes[0, i].imshow(np.clip(img_disp, 0, 1)); axes[0, i].set_axis_off(); axes[0, i].set_title(base)
        axes[1, i].imshow(heat_vis_resized, cmap="hot"); axes[1, i].set_axis_off();
    plt.tight_layout()
    grid_path = os.path.join(LOG_DIR, "sanity_grid.png")
    os.makedirs(LOG_DIR, exist_ok=True)
    plt.savefig(grid_path, dpi=120)
    plt.show()
    # plt.close(fig)
    print(f"Precheck grid saved → {grid_path}")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Preparing datasets from {IMG_DIR} with annotations in {JSON_DIR}")
    print(f"NUM_V loaded from mesh info: {NUM_V}")
    top, left = get_center_crop_offsets()
    print(f"Padded canvas: ({PADDED_HEIGHT}x{PADDED_WIDTH})")
    print(f"Center crop (top,left,height,width) = ({top}, {left}, {CROP_HEIGHT}, {CROP_WIDTH})")
    visualize_precheck_grid(num_samples=4)
    write_npy_datasets()