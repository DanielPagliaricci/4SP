#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


#############################
# CONFIGURATIONS #
#############################
# TODO:  remove cube topology hardcoded
DEBUG = True

##### VISUALIZATION CONTROLS #####
GENERATE_SAMPLE_VIZ = True  # set False to skip creating the sample visualization
SHOW_SAMPLE_VIZ = True      # set False to save without displaying the window

# Number of augmented samples to generate per real data sample
AMOUNT_DATA_PER_REAL_DATA = 5

# Image size used for normalization/denormalization
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(BASE_DIR)

VERTICES_DATA_FOLDER = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "JSON_Data_REF")
IMAGES_DATA_FOLDER = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Rendered_Images")

OBJECT_MESH_INFO = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Object_Information", "mesh_information_REF_TANGO.json")

# Load number of vertices from mesh info (same approach as visible preprocess)
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

NUM_V = load_num_vertices(OBJECT_MESH_INFO, default_num_vertices=8)

# Augmentation parameters (same semantics as in train_invisible_robust.py)
AUGMENTATION_FLIP_PROBABILITY = 0.2#0.1
AUGMENTATION_POSITIONAL_NOISE_STDDEV_PIXELS = 3#1.5
AUGMENTATION_MAX_FLIPPED_VERTICES = 2  # None = unlimited

# Output paths (saved next to this script by default)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
X_OUTPUT_PATH = os.path.join(THIS_DIR, "X_data.npy")
Y_OUTPUT_PATH = os.path.join(THIS_DIR, "Y_data.npy")
SAMPLE_VIZ_PATH = os.path.join(THIS_DIR, "sample_preprocess_viz.png")

# Reproducibility
SEED = 42
np.random.seed(SEED)


# =============================
# Helper functions (mirroring train_invisible_robust.py)
# =============================
def polygon_area(coords: np.ndarray) -> float:
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def compute_hybrid_features(x: np.ndarray,
                            edges,
                            faces,
                            image_width: int = IMAGE_WIDTH,
                            image_height: int = IMAGE_HEIGHT,
                            num_vertices: int = 8) -> np.ndarray:
    """
    Compute hybrid features from raw vertex data x (flattened num_vertices×3 array).

    Layout:
    - First 3*num_vertices: vertex data (flag, norm_x, norm_y).
    - Next 3*len(edges): per-edge features [distance_norm, cos(theta), sin(theta)] if both vertices are visible, else zeros.
      Distance is computed in pixel space and normalized by the image diagonal.
      Angle is computed with arctan2 in pixel space to avoid aspect-ratio bias; we encode as (cos, sin) to avoid wrap-around.
    - Next 2*len(faces): face centroid (norm_x, norm_y) if all face vertices are visible, else zeros.
    - Next len(faces): normalized projected area per face (pixel area / image area), else zero.
    - Final 3: convex hull features of visible (non-flipped) vertices: [normalized_area, normalized_perimeter, normalized_vertex_count].
    """
    v_pixels = x.reshape(num_vertices, 3).copy() if x.ndim == 1 else x.copy()
    v_normalized = v_pixels.copy()

    # Normalize pixel coordinates
    v_normalized[:, 1] = v_normalized[:, 1] / image_width
    v_normalized[:, 2] = v_normalized[:, 2] / image_height

    # Visible-only coordinates to avoid leakage: zero coords where flag == 0
    v_feats = v_normalized.copy()
    v_feats[v_feats[:, 0] == 0.0, 1:3] = 0.0
    vertex_features = v_feats.flatten()

    # Edge features: for each edge we add [distance_norm, cos(theta), sin(theta)]
    edge_features = []
    image_diagonal = float(np.hypot(image_width, image_height))
    for (i, j) in edges:
        if v_normalized[i, 0] == 1.0 and v_normalized[j, 0] == 1.0:
            dx_px = float(v_pixels[i, 1] - v_pixels[j, 1])
            dy_px = float(v_pixels[i, 2] - v_pixels[j, 2])
            distance_norm = (np.hypot(dx_px, dy_px) / image_diagonal) if image_diagonal > 0.0 else 0.0
            angle = np.arctan2(dy_px, dx_px)
            edge_features.extend([distance_norm, np.cos(angle), np.sin(angle)])
        else:
            edge_features.extend([0.0, 0.0, 0.0])
    edge_features = np.array(edge_features, dtype=np.float32)

    # Face centroid and area features
    face_centroid_features = []
    face_area_features = []
    total_image_area = image_width * image_height

    if faces is not None:
        for face_indices in faces:
            are_all_vertices_visible = all(v_normalized[idx, 0] == 1.0 for idx in face_indices)
            if are_all_vertices_visible:
                face_vertices_norm_coords = v_normalized[face_indices, 1:3]
                centroid = np.mean(face_vertices_norm_coords, axis=0)
                face_centroid_features.extend(centroid)

                face_vertices_pixel_coords = v_pixels[face_indices, 1:3]
                pixel_area = polygon_area(face_vertices_pixel_coords)
                normalized_area = pixel_area / total_image_area
                face_area_features.append(normalized_area)
            else:
                face_centroid_features.extend([0.0, 0.0])
                face_area_features.append(0.0)
    face_centroid_features = np.array(face_centroid_features, dtype=np.float32)
    face_area_features = np.array(face_area_features, dtype=np.float32)

    # Convex hull features
    visible_vertices_pixels = v_pixels[v_pixels[:, 0] == 1.0][:, 1:3]
    hull_area = 0.0
    hull_perimeter = 0.0
    hull_num_vertices = 0.0
    if len(visible_vertices_pixels) >= 3:
        try:
            hull = ConvexHull(visible_vertices_pixels)
            hull_area = hull.area
            hull_perimeter = hull.volume  # perimeter in 2D
            hull_num_vertices = len(hull.vertices)
        except Exception:
            pass

    image_diagonal = np.sqrt(image_width ** 2 + image_height ** 2)
    normalized_hull_area = hull_area / (image_width * image_height)
    normalized_hull_perimeter = hull_perimeter / image_diagonal if image_diagonal > 0 else 0.0
    normalized_hull_num_vertices = hull_num_vertices / float(max(1, num_vertices))

    hull_features = np.array([
        normalized_hull_area,
        normalized_hull_perimeter,
        normalized_hull_num_vertices,
    ], dtype=np.float32)

    hybrid = np.concatenate([
        vertex_features,
        edge_features,
        face_centroid_features,
        face_area_features,
        hull_features,
    ])
    return hybrid


def _load_vertex_data_from_json(json_path: str,
                                num_vertices: int,
                                image_width: int = IMAGE_WIDTH,
                                image_height: int = IMAGE_HEIGHT):
    """
    Load vertex data and targets from JSON (no augmentation here).
    Returns:
      - initial_full_data: (num_vertices, 3) [flag, pixel_x, pixel_y]
      - initial_target: (num_vertices, 5) [mask, norm_x, norm_y, original_visibility_flag, augmented_flip_flag=0]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    initial_full_data = np.zeros((num_vertices, 3), dtype=np.float32)
    initial_target = np.zeros((num_vertices, 5), dtype=np.float32)

    processed_indices = set()

    for vertex in data.get('visible_vertices', []):
        idx = vertex['index']
        if idx >= num_vertices:
            continue
        if idx in processed_indices:
            if DEBUG:
                print(f"Warning: Vertex {idx} in {json_path} already processed (visible).")
            continue
        x_coord, y_coord = vertex['pixel_coordinates']
        initial_full_data[idx] = [1.0, x_coord, y_coord]
        initial_target[idx] = [0.0, x_coord / image_width, y_coord / image_height, 1.0, 0.0]
        processed_indices.add(idx)

    for vertex in data.get('non_visible_vertices', []):
        idx = vertex['index']
        if idx >= num_vertices:
            continue
        x_coord, y_coord = vertex['pixel_coordinates']
        initial_full_data[idx] = [0.0, x_coord, y_coord]
        initial_target[idx] = [1.0, x_coord / image_width, y_coord / image_height, 0.0, 0.0]
        processed_indices.add(idx)

    return initial_full_data, initial_target


def augment_and_feature_engineer_sample_numpy(initial_full_data: np.ndarray,
                                              initial_target: np.ndarray,
                                              flip_prob: float,
                                              noise_stddev: float,
                                              edges,
                                              faces,
                                              image_width: int,
                                              image_height: int,
                                              num_vertices: int):
    # Copies to modify
    full_data_for_features = initial_full_data.copy()
    augmented_target = initial_target.copy()

    # 1) Random flip of originally visible vertices to non-visible
    if flip_prob > 0:
        visible_indices = [idx for idx in range(num_vertices) if initial_target[idx, 3] == 1.0]
        flip_candidates = []
        for idx in visible_indices:
            if np.random.rand() < flip_prob:
                flip_candidates.append(idx)
        max_flips = AUGMENTATION_MAX_FLIPPED_VERTICES
        if max_flips is not None and len(flip_candidates) > max_flips:
            flip_candidates = list(np.random.choice(flip_candidates, max_flips, replace=False))
        for idx in flip_candidates:
            full_data_for_features[idx, 0] = 0.0
            augmented_target[idx, 0] = 1.0
            augmented_target[idx, 4] = 1.0  # augmented_flip_flag

    # 2) Positional noise added to pixel coords used for feature computation
    if noise_stddev > 0:
        noise = np.random.normal(0, noise_stddev, size=(num_vertices, 2)).astype(np.float32)
        full_data_for_features[:, 1:3] += noise
        full_data_for_features[:, 1] = np.clip(full_data_for_features[:, 1], 0, image_width - 1)
        full_data_for_features[:, 2] = np.clip(full_data_for_features[:, 2], 0, image_height - 1)

    # 3) Feature engineering
    hybrid_features = compute_hybrid_features(full_data_for_features, edges, faces, image_width, image_height, num_vertices)
    noised_pixel_coords = full_data_for_features[:, 1:3].copy()

    return hybrid_features.astype(np.float32), augmented_target.astype(np.float32), noised_pixel_coords.astype(np.float32)


def visualize_sample_data(image_file_path: str,
                          original_vertices_pixel_coords: np.ndarray,
                          target_vertex_info: np.ndarray,
                          edges,
                          faces,
                          output_path: str,
                          image_width: int,
                          image_height: int,
                          num_vertices: int,
                          noised_vertices_pixel_coords: np.ndarray = None,
                          noise_stddev_active: float = 0.0,
                          show: bool = True):
    try:
        img = plt.imread(image_file_path)
    except Exception:
        if DEBUG:
            print(f"Warning: Could not load image at {image_file_path}. Using a placeholder.")
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    fig, ax = plt.subplots(1, figsize=(12, 12 * (img.shape[0] / img.shape[1]) if img.shape[1] > 0 else 12))
    ax.imshow(img)

    colors = {
        "visible_original": "green",
        "nonvisible_original": "red",
        "visible_flipped": "orange",
    }
    legend_handles = {}

    coords_for_drawing = noised_vertices_pixel_coords if noised_vertices_pixel_coords is not None else original_vertices_pixel_coords

    # Convex hull for visible (original and not flipped)
    visible_for_features_mask = np.logical_and(target_vertex_info[:, 3] == 1.0, target_vertex_info[:, 4] == 0.0)
    visible_coords_for_hull_calc = coords_for_drawing[visible_for_features_mask]
    if len(visible_coords_for_hull_calc) >= 3:
        try:
            hull = ConvexHull(visible_coords_for_hull_calc)
            for simplex in hull.simplices:
                ax.plot(visible_coords_for_hull_calc[simplex, 0], visible_coords_for_hull_calc[simplex, 1], 'w--', zorder=1.5)
        except Exception:
            pass

    # Draw edges
    for u, v in edges:
        u_vis = (target_vertex_info[u, 3] == 1.0 and target_vertex_info[u, 4] == 0.0)
        v_vis = (target_vertex_info[v, 3] == 1.0 and target_vertex_info[v, 4] == 0.0)
        start_pos = coords_for_drawing[u]
        end_pos = coords_for_drawing[v]
        if u_vis and v_vis:
            line, = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='cyan', linestyle='--', linewidth=2, alpha=0.9, zorder=1)
            legend_handles.setdefault("Active Edge (in features)", line)
        else:
            line, = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
            legend_handles.setdefault("Inactive Edge", line)

    # Face centroids
    if faces is not None:
        for face_indices in faces:
            is_face_visible = all((target_vertex_info[idx, 3] == 1.0 and target_vertex_info[idx, 4] == 0.0) for idx in face_indices)
            if is_face_visible:
                face_vertex_coords = coords_for_drawing[face_indices]
                centroid = np.mean(face_vertex_coords, axis=0)
                handle = ax.scatter(centroid[0], centroid[1], color='yellow', marker='*', s=150, edgecolors='black', zorder=5)
                legend_handles.setdefault("Visible Face Centroid", handle)

    # Draw vertices
    for j in range(num_vertices):
        px_orig, py_orig = original_vertices_pixel_coords[j, 0], original_vertices_pixel_coords[j, 1]
        orig_visible = target_vertex_info[j, 3] == 1.0
        is_flipped = target_vertex_info[j, 4] == 1.0
        if orig_visible and is_flipped:
            label = "Visible (Flipped to Non-Vis)"
            color = colors["visible_flipped"]
        elif orig_visible and not is_flipped:
            label = "Visible (Original)"
            color = colors["visible_original"]
        else:
            label = "Non-Visible (Original)"
            color = colors["nonvisible_original"]

        handle = ax.scatter(px_orig, py_orig, color=color, s=100, edgecolors='black', alpha=0.8, zorder=3)
        legend_handles.setdefault(label, handle)
        ax.text(px_orig + 15, py_orig, str(j), color='white', fontsize=10, ha='left', va='center',
                bbox=dict(facecolor=color, alpha=0.7, pad=0.2, edgecolor='none'), zorder=4)

        if noised_vertices_pixel_coords is not None and noise_stddev_active > 0.1:
            px_noised, py_noised = noised_vertices_pixel_coords[j, 0], noised_vertices_pixel_coords[j, 1]
            if abs(px_orig - px_noised) > 0.1 or abs(py_orig - py_noised) > 0.1:
                ax.plot([px_orig, px_noised], [py_orig, py_noised], color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
                ax.scatter(px_noised, py_noised, color=color, s=30, marker='x', alpha=0.7, zorder=3)

    ax.legend(legend_handles.values(), legend_handles.keys(), title="Vertex & Edge Status", loc="upper right")
    ax.set_title("Sample Preprocess Visualization")
    ax.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        if DEBUG:
            print(f"Saved sample visualization to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")
    if show:
        plt.show()
    plt.close(fig)


# =============================
# Main preprocess
# =============================
def main():
    # Load mesh info for edges and faces
    num_vertices = load_num_vertices(OBJECT_MESH_INFO, default_num_vertices=NUM_V)

    try:
        with open(OBJECT_MESH_INFO, 'r') as f:
            mesh_info = json.load(f)
        edges = mesh_info.get('edge_composition')
        faces = mesh_info.get('face_composition')
        if edges is None or faces is None:
            raise ValueError("edge_composition or face_composition missing in mesh info")
        if DEBUG:
            print(f"Loaded edges ({len(edges)}) and faces ({len(faces)}) from {OBJECT_MESH_INFO}")
    except Exception as e:
        print(f"Warning: Could not load mesh info: {e}. Falling back to hardcoded cube topology.")
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        faces = [
            (0, 4, 6, 2), (3, 2, 6, 7), (7, 6, 4, 5),
            (5, 1, 3, 7), (1, 0, 2, 3), (5, 4, 0, 1),
        ]
        num_vertices = 8

    # Collect JSON files
    if not os.path.isdir(VERTICES_DATA_FOLDER):
        print(f"Error: VERTICES_DATA_FOLDER not found: {VERTICES_DATA_FOLDER}")
        return

    json_files = sorted([
        os.path.join(VERTICES_DATA_FOLDER, f)
        for f in os.listdir(VERTICES_DATA_FOLDER)
        if f.endswith('.json')
    ])

    if not json_files:
        print("No JSON files found to preprocess.")
        return

    if DEBUG:
        print(f"Found {len(json_files)} JSON files. Starting preprocess...")

    X_list = []  # (N, 69)
    Y_list = []  # (N, 8, 5)

    # Iterate and preprocess
    for idx, json_path in enumerate(json_files):
        initial_full, initial_target = _load_vertex_data_from_json(json_path, num_vertices, IMAGE_WIDTH, IMAGE_HEIGHT)
        # Generate multiple augmented variants per real data sample
        for _ in range(AMOUNT_DATA_PER_REAL_DATA):
            hybrid, aug_target, _noised = augment_and_feature_engineer_sample_numpy(
                initial_full,
                initial_target,
                AUGMENTATION_FLIP_PROBABILITY,
                AUGMENTATION_POSITIONAL_NOISE_STDDEV_PIXELS,
                edges,
                faces,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                num_vertices,
            )
            X_list.append(hybrid)
            Y_list.append(aug_target)

        if DEBUG and (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(json_files)} files...")

    X_data = np.stack(X_list, axis=0)
    Y_data = np.stack(Y_list, axis=0)

    # Save outputs
    try:
        np.save(X_OUTPUT_PATH, X_data)
        np.save(Y_OUTPUT_PATH, Y_data)
        print(f"Saved X_data to {X_OUTPUT_PATH} with shape {X_data.shape}")
        print(f"Saved Y_data to {Y_OUTPUT_PATH} with shape {Y_data.shape}")
    except Exception as e:
        print(f"Error saving .npy files: {e}")

    if GENERATE_SAMPLE_VIZ:
        # Plot a sample visualization for sanity check
        # Pick the first sample for reproducibility
        sample_json = json_files[0]
        base_name = os.path.splitext(os.path.basename(sample_json))[0]
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(IMAGES_DATA_FOLDER, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        # Recompute augmentation for the sample to obtain noised coords and target used for features
        initial_full, initial_target = _load_vertex_data_from_json(sample_json, num_vertices, IMAGE_WIDTH, IMAGE_HEIGHT)
        _, target_for_viz, noised_coords = augment_and_feature_engineer_sample_numpy(
            initial_full,
            initial_target,
            AUGMENTATION_FLIP_PROBABILITY,
            AUGMENTATION_POSITIONAL_NOISE_STDDEV_PIXELS,
            edges,
            faces,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            num_vertices,
        )
        original_coords = initial_full[:, 1:3]

        if image_path is None:
            if DEBUG:
                print(f"Could not find an image for {base_name} in {IMAGES_DATA_FOLDER}. A placeholder will be shown.")
            image_path = os.path.join(IMAGES_DATA_FOLDER, base_name + '.png')  # will fail and fallback to placeholder

        visualize_sample_data(
            image_file_path=image_path,
            original_vertices_pixel_coords=original_coords,
            target_vertex_info=target_for_viz,
            edges=edges,
            faces=faces,
            output_path=SAMPLE_VIZ_PATH,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            num_vertices=num_vertices,
            noised_vertices_pixel_coords=noised_coords,
            noise_stddev_active=AUGMENTATION_POSITIONAL_NOISE_STDDEV_PIXELS,
            show=SHOW_SAMPLE_VIZ,
        )


if __name__ == "__main__":
    main()


