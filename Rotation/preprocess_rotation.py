#!/usr/bin/env python3
# TANGO_preprocess_data_noise.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import load_vertex_data, compute_hybrid_features_all_vertices, load_quaternion_from_json, load_mesh_geometry

# --- Configuration Constants ---
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
BATCH_SIZE = 100000  # Process files in batches of this size
AMOUNT_DATA_PER_REAL_DATA = 10#15  # Number of augmented samples per JSON

# --- Noise Hyperparameters ---
# Small noise is applied to all vertices; big noise is applied to a subset
# of vertices controlled by BIG_NOISE_APPLY_TO_VISIBLE (True => visible, False => non-visible)
NOISE_SMALL_PX = 8.0
NOISE_BIG_PX = 10.0
NOISE_MAX_BIG_VERTICES = 4
BIG_NOISE_APPLY_TO_VISIBLE = False

BASE_DATA_DIR = "/Users/dani/Desktop/ICTati/Second_Attempt_AngularDataBase/TANGO"
DATA_DIR = os.path.join(BASE_DATA_DIR, "1M")
#DATA_DIR = os.path.join(BASE_DATA_DIR, "vertices_data_sample_TANGO_REF_sample1")
OBJECT_MESH_PATH = os.path.join(BASE_DATA_DIR, "Object_Information", "mesh_information_REF_TANGO.json")

# Use a separate output directory for noised data
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves_vastai_noise_not_visible_1M")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "data_batch")

# --- Visualization Options ---
ENABLE_OVERLAY_VISUALIZATION = False  # Set to True to save overlays
MAX_OVERLAY_SAMPLES = 10            # Maximum number of overlays to save across the run
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "renders_sample_TANGO_sample1")  # Folder containing images
OVERLAYS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_inspection")

if not os.path.exists(OVERLAYS_DIR) and ENABLE_OVERLAY_VISUALIZATION:   
    os.makedirs(OVERLAYS_DIR, exist_ok=True)


def _find_image_for_json(json_path: str, images_dir: str) -> str:
    base = os.path.splitext(os.path.basename(json_path))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return ""


def _save_overlay(image_path: str, vertices_flat: np.ndarray, out_path: str, image_width: int, image_height: int) -> None:
    try:
        img = plt.imread(image_path)
    except Exception:
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    num_vertices = int(vertices_flat.size // 3)
    verts = vertices_flat.reshape(num_vertices, 3)

    fig, ax = plt.subplots(1, figsize=(12, 12 * (img.shape[0] / img.shape[1]) if img.shape[1] > 0 else 12))
    ax.imshow(img)

    legend_handles = {}
    for i in range(num_vertices):
        is_visible = (verts[i, 0] == 1.0)
        px, py = float(verts[i, 1]), float(verts[i, 2])
        color = 'seagreen' if is_visible else 'khaki'
        label = 'Visible Vertex' if is_visible else 'Non-Visible Vertex'
        scatter = ax.scatter(px, py, c=color, s=80, edgecolors='black', zorder=3, alpha=0.9)
        if label not in legend_handles:
            legend_handles[label] = scatter
        ax.text(px + 8, py, str(i), color='white', fontsize=9,
                bbox=dict(facecolor=color, alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.25'), zorder=4)

    if legend_handles:
        ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title="Key", loc="upper right")
    ax.set_title(f"Noised vertices overlay: {os.path.basename(image_path)}", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _save_overlay_multi(image_path: str, list_vertices_flat: list, out_path: str, image_width: int, image_height: int) -> None:
    try:
        img = plt.imread(image_path)
    except Exception:
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    fig, ax = plt.subplots(1, figsize=(12, 12 * (img.shape[0] / img.shape[1]) if img.shape[1] > 0 else 12))
    ax.imshow(img)

    # Distinct colors and small alphas for different augmentations
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
    alpha_visible = 0.60
    alpha_non_visible = 0.35

    legend_handles = {}

    for aug_idx, vertices_flat in enumerate(list_vertices_flat):
        color_aug = palette[aug_idx % len(palette)]
        num_vertices = int(vertices_flat.size // 3)
        verts = vertices_flat.reshape(num_vertices, 3)
        for i in range(num_vertices):
            is_visible = (verts[i, 0] == 1.0)
            px, py = float(verts[i, 1]), float(verts[i, 2])
            # Use lighter alpha for non-visible to keep background readable
            alpha_point = alpha_visible if is_visible else alpha_non_visible
            scatter = ax.scatter(px, py, c=color_aug, s=36, edgecolors='none', zorder=3, alpha=alpha_point)
            label = f'Aug {aug_idx+1}'
            if label not in legend_handles:
                legend_handles[label] = scatter

    if legend_handles:
        ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title="Augmentations", loc="upper right")
    ax.set_title(f"Noised vertices overlay (multi-aug): {os.path.basename(image_path)}", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def add_pixel_noise_to_vertices(
    vertex_flattened: np.ndarray,
    image_width: int,
    image_height: int,
    max_big_vertices: int = 11,
    big_noise_px: float = 0.0,
    small_noise_px: float = 10.0,
    prefer_visible: bool = True,
) -> np.ndarray:
    """
    Add pixel noise to vertex coordinates.

    - Up to `max_big_vertices` get larger noise (±big_noise_px).
    - Larger noise is applied only to non-visible vertices (v[:, 0] != 1.0).
    - All other vertices get smaller noise (±small_noise_px).
    - Only pixel coordinates (x, y) are perturbed; visibility flags are preserved.
    - Coordinates are clamped to image bounds after noise.
    """
    if vertex_flattened.size == 0:
        return vertex_flattened

    num_vertices = int(vertex_flattened.size // 3)
    v = vertex_flattened.reshape(num_vertices, 3).copy()

    # Indices eligible for big noise: choose set based on visibility
    if prefer_visible:
        candidate_indices = np.where(v[:, 0] == 1.0)[0]
    else:
        candidate_indices = np.where(v[:, 0] != 1.0)[0]

    if candidate_indices.size > 0:
        num_big = int(min(max_big_vertices, candidate_indices.size))
        big_indices = np.random.choice(candidate_indices, size=num_big, replace=False)
    else:
        big_indices = np.array([], dtype=int)

    # Start with small noise for all vertices
    small_dx = np.random.uniform(-small_noise_px, small_noise_px, size=num_vertices)
    small_dy = np.random.uniform(-small_noise_px, small_noise_px, size=num_vertices)

    dx = small_dx
    dy = small_dy

    # Override selected vertices with big noise
    if big_indices.size > 0:
        dx_big = np.random.uniform(-big_noise_px, big_noise_px, size=big_indices.size)
        dy_big = np.random.uniform(-big_noise_px, big_noise_px, size=big_indices.size)
        dx[big_indices] = dx_big
        dy[big_indices] = dy_big

    # Apply and clamp
    v[:, 1] = np.clip(v[:, 1] + dx, 0.0, float(image_width - 1))
    v[:, 2] = np.clip(v[:, 2] + dy, 0.0, float(image_height - 1))

    return v.astype(np.float32).flatten()


def main():
    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = DATA_DIR

    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} already exists. Please delete it or choose a different output directory.")
        #return
    else:
        os.makedirs(OUTPUT_DIR)

    try:
        edges, faces, total_vertices = load_mesh_geometry(OBJECT_MESH_PATH)
        if not faces:
            print("Warning: 'face_composition' missing or empty in mesh_information.json; proceeding with empty faces list.")
    except Exception as e:
        print(f"Error loading mesh information from {OBJECT_MESH_PATH}: {e}")
        return

    print(f"Data directory: {data_dir}")
    try:
        json_files_generator = (
            os.path.join(data_dir, f.name)
            for f in os.scandir(data_dir)
            if f.name.endswith(".json") and f.is_file()
        )
    except FileNotFoundError:
        print(f"Error: Data directory not found at {data_dir}")
        print("Please ensure your data is located in the configured DATA_DIR.")
        return

    X_list = []
    Y_list = []
    batch_num = 1
    files_processed = 0
    samples_processed = 0
    print("Starting file processing with noise injection...")

    overlays_saved = 0

    for jf in json_files_generator:
        try:
            raw_vertex = load_vertex_data(jf, num_vertices=total_vertices if total_vertices > 0 else None)

            # Generate multiple augmented variants per real data sample
            aug_vertices_accum = [] if ENABLE_OVERLAY_VISUALIZATION else None
            for aug_idx in range(AMOUNT_DATA_PER_REAL_DATA):
                noisy_vertex = add_pixel_noise_to_vertices(
                    raw_vertex,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                    max_big_vertices=NOISE_MAX_BIG_VERTICES,
                    big_noise_px=NOISE_BIG_PX,
                    small_noise_px=NOISE_SMALL_PX,
                    prefer_visible=BIG_NOISE_APPLY_TO_VISIBLE,
                )

                # Use all-vertices features to train with maximum available information
                hybrid = compute_hybrid_features_all_vertices(noisy_vertex, edges, faces, IMAGE_WIDTH, IMAGE_HEIGHT)
                q = load_quaternion_from_json(jf)
                X_list.append(hybrid)
                Y_list.append(q)
                samples_processed += 1

                # Accumulate per-JSON augmentations for a single overlay image
                if ENABLE_OVERLAY_VISUALIZATION:
                    aug_vertices_accum.append(noisy_vertex)

            # After generating all augmentations for this JSON, save one combined overlay (capped)
            if ENABLE_OVERLAY_VISUALIZATION and overlays_saved < MAX_OVERLAY_SAMPLES:
                img_path = _find_image_for_json(jf, IMAGES_DIR)
                if img_path and aug_vertices_accum:
                    base_name = os.path.splitext(os.path.basename(jf))[0]
                    out_viz = os.path.join(OVERLAYS_DIR, f"prediction_viz_rotation_{base_name}.png")
                    try:
                        _save_overlay_multi(
                            image_path=img_path,
                            list_vertices_flat=aug_vertices_accum,
                            out_path=out_viz,
                            image_width=IMAGE_WIDTH,
                            image_height=IMAGE_HEIGHT,
                        )
                        overlays_saved += 1
                    except Exception as viz_e:
                        print(f"Failed to save overlay for {jf}: {viz_e}")

                # Save data in batches (by augmented sample count)
                if samples_processed % BATCH_SIZE == 0:
                    X = np.array(X_list, dtype=np.float32)
                    Y = np.array(Y_list, dtype=np.float32)

                    print(f"\nSaving batch {batch_num}. Shapes are:")
                    print(f"  X: {X.shape}")
                    print(f"  Y: {Y.shape}")

                    output_x_path = os.path.join(OUTPUT_DIR, f"X_data_batch_{batch_num}.npy")
                    output_y_path = os.path.join(OUTPUT_DIR, f"Y_data_batch_{batch_num}.npy")

                    np.save(output_x_path, X)
                    np.save(output_y_path, Y)

                    print(f"Successfully saved batch feature data to {output_x_path}")
                    print(f"Successfully saved batch label data to {output_y_path}")

                    X_list = []
                    Y_list = []
                    batch_num += 1

            files_processed += 1

            # Periodic progress by files
            if files_processed % 500 == 0:
                print(f"  Processed {files_processed} files... (generated {samples_processed} samples)")
        except Exception as e:
            print(f"Skipping {jf} due to error: {e}")
            continue

    # Save the final batch if any files are remaining
    if X_list:
        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)

        print(f"\nSaving final batch {batch_num}. Shapes are:")
        print(f"  X: {X.shape}")
        print(f"  Y: {Y.shape}")

        output_x_path = os.path.join(OUTPUT_DIR, f"X_data_batch_{batch_num}.npy")
        output_y_path = os.path.join(OUTPUT_DIR, f"Y_data_batch_{batch_num}.npy")

        np.save(output_x_path, X)
        np.save(output_y_path, Y)

        print(f"Successfully saved final batch feature data to {output_x_path}")
        print(f"Successfully saved final batch label data to {output_y_path}")

    print(f"\nAll batches processed with noise. Total files: {files_processed}. Total samples: {samples_processed}.")


if __name__ == "__main__":
    main()

