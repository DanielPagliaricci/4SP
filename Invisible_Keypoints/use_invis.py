#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from preprocess_invis import (
    compute_hybrid_features,
    _load_vertex_data_from_json,
    load_num_vertices,
    visualize_sample_data,
    augment_and_feature_engineer_sample_numpy,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)


#############################
# CONFIGURATIONS #
#############################

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(BASE_DIR)
VERTICES_DATA_FOLDER: str = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "JSON_Data_REF")
IMAGES_DATA_FOLDER: str = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Rendered_Images")
OBJECT_MESH_INFO = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Object_Information", "mesh_information_REF_TANGO.json")

MODEL_PATH = os.path.join(BASE_DIR, "BestModel", "best_model_invisible.h5")
LOG_DIR = os.path.join(BASE_DIR, "Use_Saved_Plots_speed")

ANALYSIS_NUM_SAMPLES = 10
TTA_ENABLE = True
TTA_SAMPLES = 8
TTA_NOISE_STDDEV_PIXELS = 1.5

# Visualization crop size (center crop)
CROP_SIZE = 1100

# Toggle background image display
SHOW_IMAGE_BACKGROUND = True


##### UTILITIES #####
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_mesh_topology(info_path):
    try:
        with open(info_path, 'r') as f:
            mesh_info = json.load(f)
        edges = mesh_info.get('edge_composition')
        faces = mesh_info.get('face_composition')
        if edges is None or faces is None:
            raise ValueError("edge_composition or face_composition missing in mesh info")
        return edges, faces
    except Exception as e:
        print(f"Warning: Could not load mesh info: {e}. Falling back to cube topology.")
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        faces = [
            (0, 4, 6, 2), (3, 2, 6, 7), (7, 6, 4, 5),
            (5, 1, 3, 7), (1, 0, 2, 3), (5, 4, 0, 1),
        ]
        return edges, faces


def visualize_feature_vector(features: np.ndarray, edges, faces, num_vertices: int, output_path: str):
    try:
        total_len = features.shape[0]
        num_edges = len(edges) if edges is not None else 0
        num_faces = len(faces) if faces is not None else 0

        seg_vertex = 3 * num_vertices
        seg_edges = 3 * num_edges
        seg_face_centroids = 2 * num_faces
        seg_face_areas = num_faces
        seg_hull = 3

        idx0 = 0
        idx1 = idx0 + seg_vertex
        idx2 = idx1 + seg_edges
        idx3 = idx2 + seg_face_centroids
        idx4 = idx3 + seg_face_areas
        idx5 = idx4 + seg_hull

        x = np.arange(total_len)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x, features, color='steelblue', width=0.8)

        # Segment boundaries
        for xi in [idx1 - 0.5, idx2 - 0.5, idx3 - 0.5, idx4 - 0.5]:
            ax.axvline(x=xi, color='black', linestyle='--', linewidth=1)

        # Labels centered per segment
        def mid(a, b):
            return (a + b - 1) / 2.0

        ax.text(mid(idx0, idx1), 1.02, 'Vertices (flag,x,y)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
        if seg_edges > 0:
            ax.text(mid(idx1, idx2), 1.02, 'Edges (dist,cos,sin)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
        if seg_face_centroids > 0:
            ax.text(mid(idx2, idx3), 1.02, 'Face centroids (x,y)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
        if seg_face_areas > 0:
            ax.text(mid(idx3, idx4), 1.02, 'Face areas', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)
        ax.text(mid(idx4, min(idx5, total_len)), 1.02, 'Hull (area,perim,count)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=9)

        ax.set_xlim(-1, max(total_len, 1))
        ax.set_title('Hybrid feature vector (model input)')
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Value')
        fig.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"Error visualizing feature vector to {output_path}: {e}")
    finally:
        plt.close('all')


def _center_crop_image(img: np.ndarray, crop_size: int = CROP_SIZE):
    h, w = img.shape[0], img.shape[1]
    cx = w // 2
    cy = h // 2
    half = crop_size // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        shift = x1 - w
        x0 = max(0, x0 - shift)
        x1 = w
    if y1 > h:
        shift = y1 - h
        y0 = max(0, y0 - shift)
        y1 = h
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(w, x1))
    y1 = int(min(h, y1))
    crop = img[y0:y1, x0:x1]
    return crop, x0, y0


def _face_color(i: int):
    cmap = plt.get_cmap('tab20')
    return cmap(i % 20)


from PIL import Image

def visualize_input_overlay(image_path: str,
                            original_vertices_pixel_coords: np.ndarray,
                            target_vertex_info: np.ndarray,
                            edges,
                            faces,
                            output_path: str,
                            image_width: int,
                            image_height: int,
                            num_vertices: int):
    try:
        # Load and force resize to expected dimensions to ensure alignment
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((image_width, image_height), Image.LANCZOS)
        img = np.array(pil_img)
    except Exception:
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    cropped, off_x, off_y = _center_crop_image(img, CROP_SIZE)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    if SHOW_IMAGE_BACKGROUND:
        ax.imshow(cropped)
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    ax.set_xlim(0, cropped.shape[1])
    ax.set_ylim(cropped.shape[0], 0)
    ax.set_aspect('equal')

    visible_mask = (target_vertex_info[:, 3] == 1.0)

    # Faces: only those fully visible
    if faces is not None:
        for fi, face_indices in enumerate(faces):
            if all(visible_mask[idx] for idx in face_indices):
                pts = original_vertices_pixel_coords[face_indices]
                pts_shift = pts - np.array([off_x, off_y])
                poly = Polygon(pts_shift, closed=True, facecolor=_face_color(fi), alpha=0.18, edgecolor='none', zorder=1)
                ax.add_patch(poly)
                centroid = np.mean(pts_shift, axis=0)
                ax.scatter(centroid[0], centroid[1], color='yellow', marker='*', s=120, edgecolors='black', linewidths=0.5, zorder=3)

    # Edges: only where both ends visible
    for u, v in edges:
        if visible_mask[u] and visible_mask[v]:
            p0 = original_vertices_pixel_coords[u] - np.array([off_x, off_y])
            p1 = original_vertices_pixel_coords[v] - np.array([off_x, off_y])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='cyan', linestyle='--', linewidth=1.5, alpha=0.9, zorder=2)

    # Vertices: only visible
    for i in range(num_vertices):
        if not visible_mask[i]:
            continue
        px, py = original_vertices_pixel_coords[i] - np.array([off_x, off_y])
        ax.scatter(px, py, c='seagreen', s=80, edgecolors='black', linewidths=0.5, zorder=4)
        ax.text(px + 8, py, str(i), color='white', fontsize=9,
                bbox=dict(facecolor='seagreen', alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.25'), zorder=5)

    ax.set_title('Input to model (visible-only)')
    ax.axis('off')
    fig.tight_layout()
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"Error saving input overlay to {output_path}: {e}")
    plt.close(fig)


def visualize_prediction(image_path, y_true, y_pred_norm, edges, output_path, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    try:
        # Load and force resize to expected dimensions
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((image_width, image_height), Image.LANCZOS)
        img = np.array(pil_img)
    except Exception:
        print(f"Warning: Image not found or error reading {image_path}. Creating a blank placeholder.")
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    cropped, off_x, off_y = _center_crop_image(img, CROP_SIZE)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    if SHOW_IMAGE_BACKGROUND:
        ax.imshow(cropped)
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    ax.set_xlim(0, cropped.shape[1])
    ax.set_ylim(cropped.shape[0], 0)
    ax.set_aspect('equal')

    true_coords_norm = y_true[:, 1:3]
    img_dims = np.array([image_width, image_height])
    true_coords_pixel = true_coords_norm * img_dims
    pred_coords_pixel = y_pred_norm * img_dims

    legend_handles = {}

    # Draw only edges whose endpoints were originally visible (match feature-calculation policy)
    visible_mask = (y_true[:, 3] == 1.0)
    for u, v in edges:
        if not (visible_mask[u] and visible_mask[v]):
            continue
        start_pos = true_coords_pixel[u] - np.array([off_x, off_y])
        end_pos = true_coords_pixel[v] - np.array([off_x, off_y])
        line, = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                        color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
        if "Visible Edges" not in legend_handles:
            legend_handles["Visible Edges"] = line

    num_vertices = y_true.shape[0]
    for i in range(num_vertices):
        is_originally_visible = (y_true[i, 3] == 1.0)
        if is_originally_visible:
            h = ax.scatter(true_coords_pixel[i, 0] - off_x, true_coords_pixel[i, 1] - off_y, c='seagreen', s=120, edgecolors='black', label='Visible Vertex', zorder=3, alpha=0.9)
            if "Visible Vertex" not in legend_handles:
                legend_handles["Visible Vertex"] = h
            ax.text(true_coords_pixel[i, 0] - off_x + 15, true_coords_pixel[i, 1] - off_y, str(i), color='white', fontsize=10,
                    bbox=dict(facecolor='seagreen', alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.3'), zorder=4)
        else:
            h_pred = ax.scatter(pred_coords_pixel[i, 0] - off_x, pred_coords_pixel[i, 1] - off_y, c='orange', s=120, marker='o', edgecolors='black', label='Prediction (Hidden)', zorder=3, alpha=0.9)
            if "Prediction (Hidden)" not in legend_handles:
                legend_handles["Prediction (Hidden)"] = h_pred

            h_true = ax.scatter(true_coords_pixel[i, 0] - off_x, true_coords_pixel[i, 1] - off_y, c='red', s=50, marker='x', label='Ground Truth (Hidden)', zorder=3)
            if "Ground Truth (Hidden)" not in legend_handles:
                legend_handles["Ground Truth (Hidden)"] = h_true

            ax.plot([true_coords_pixel[i, 0] - off_x, pred_coords_pixel[i, 0] - off_x],
                    [true_coords_pixel[i, 1] - off_y, pred_coords_pixel[i, 1] - off_y],
                    'r--', linewidth=1.5, alpha=0.7)

            ax.text(pred_coords_pixel[i, 0] - off_x + 15, pred_coords_pixel[i, 1] - off_y, str(i), color='white', fontsize=10,
                    bbox=dict(facecolor='orange', alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.3'), zorder=4)

    ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title="Vertex Key", loc="upper right")
    ax.set_title(f"Prediction vs Ground Truth for {os.path.basename(image_path)}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")
    plt.close(fig)


def visualize_prediction_full(image_path, y_true, y_pred_norm, edges, faces, output_path, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    try:
        # Load and force resize to expected dimensions
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize((image_width, image_height), Image.LANCZOS)
        img = np.array(pil_img)
    except Exception:
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    cropped, off_x, off_y = _center_crop_image(img, CROP_SIZE)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    if SHOW_IMAGE_BACKGROUND:
        ax.imshow(cropped)
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    ax.set_xlim(0, cropped.shape[1])
    ax.set_ylim(cropped.shape[0], 0)
    ax.set_aspect('equal')

    true_coords_norm = y_true[:, 1:3]
    img_dims = np.array([image_width, image_height])
    true_coords_pixel = true_coords_norm * img_dims
    pred_coords_pixel = y_pred_norm * img_dims

    # Mix: visible -> true, hidden -> predicted
    coords_for_edges = np.where(y_true[:, 3:4] == 1.0, true_coords_pixel, pred_coords_pixel)

    legend_handles = {}

    # Faces: fill and centroid using mixed coords
    if faces is not None:
        for fi, face_indices in enumerate(faces):
            pts = coords_for_edges[face_indices]
            pts_shift = pts - np.array([off_x, off_y])
            poly = Polygon(pts_shift, closed=True, facecolor=_face_color(fi), alpha=0.18, edgecolor='none', zorder=1)
            ax.add_patch(poly)
            centroid = np.mean(pts_shift, axis=0)
            # Use diamond for predicted centroid marker
            handle = ax.scatter(centroid[0], centroid[1], color='yellow', marker='D', s=110, edgecolors='black', linewidths=0.5, zorder=3)
            if 'Predicted Face Centroid' not in legend_handles:
                legend_handles['Predicted Face Centroid'] = handle

    # Draw all edges from mixed coords
    for u, v in edges:
        p0 = coords_for_edges[u] - np.array([off_x, off_y])
        p1 = coords_for_edges[v] - np.array([off_x, off_y])
        line, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='cyan', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
        if 'Edges (mixed)' not in legend_handles:
            legend_handles['Edges (mixed)'] = line

    # Vertices: draw hidden predictions and visible ground truth
    num_vertices = y_true.shape[0]
    for i in range(num_vertices):
        is_vis = (y_true[i, 3] == 1.0)
        if is_vis:
            h = ax.scatter(true_coords_pixel[i, 0] - off_x, true_coords_pixel[i, 1] - off_y, c='seagreen', s=90, edgecolors='black', linewidths=0.5, label='Visible Vertex', zorder=4)
            if 'Visible Vertex' not in legend_handles:
                legend_handles['Visible Vertex'] = h
        else:
            h = ax.scatter(pred_coords_pixel[i, 0] - off_x, pred_coords_pixel[i, 1] - off_y, c='orange', s=90, edgecolors='black', linewidths=0.5, label='Predicted (Hidden)', zorder=4)
            if 'Predicted (Hidden)' not in legend_handles:
                legend_handles['Predicted (Hidden)'] = h

    ax.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()), loc='upper right', fontsize='small')
    ax.set_title('Post-prediction overlay (visible GT + hidden predicted)')
    ax.axis('off')
    fig.tight_layout()
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"Error saving prediction overlay to {output_path}: {e}")
    plt.close(fig)


def main():
    ensure_dir(LOG_DIR)

    # Load topology and vertex count
    num_vertices = load_num_vertices(OBJECT_MESH_INFO, default_num_vertices=8)
    edges, faces = load_mesh_topology(OBJECT_MESH_INFO)

    # Gather dataset JSONs
    if not os.path.isdir(VERTICES_DATA_FOLDER):
        print(f"Error: VERTICES_DATA_FOLDER not found: {VERTICES_DATA_FOLDER}")
        return
    json_files = sorted([os.path.join(VERTICES_DATA_FOLDER, f) for f in os.listdir(VERTICES_DATA_FOLDER) if f.endswith('.json')])
    if not json_files:
        print("No JSON files found for inference.")
        return

    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.summary()
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        return

    # Run predictions on a subset and visualize
    num_to_run = min(ANALYSIS_NUM_SAMPLES, len(json_files))
    for i in range(num_to_run):
        jf = json_files[i]
        base_name = os.path.splitext(os.path.basename(jf))[0]
        try:
            initial_full, initial_target = _load_vertex_data_from_json(jf, num_vertices, IMAGE_WIDTH, IMAGE_HEIGHT)

            # Locate image file (needed for feature overlay visualization)
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(IMAGES_DATA_FOLDER, base_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                image_path = os.path.join(IMAGES_DATA_FOLDER, base_name + '.png')

            # Visualize input overlay (visible-only, no non-visible traces)
            try:
                input_overlay_out = os.path.join(LOG_DIR, f"input_overlay_{base_name}.png")
                visualize_input_overlay(
                    image_path=image_path,
                    original_vertices_pixel_coords=initial_full[:, 1:3],
                    target_vertex_info=initial_target,
                    edges=edges,
                    faces=faces,
                    output_path=input_overlay_out,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                    num_vertices=num_vertices,
                )
            except Exception as e_vis:
                print(f"Warning: could not create input overlay for {base_name}: {e_vis}")

            # Additionally, plot the actual feature vector layout values (non-noised baseline)
            try:
                features_for_plot = compute_hybrid_features(initial_full, edges, faces, IMAGE_WIDTH, IMAGE_HEIGHT, num_vertices)
                feature_vector_out = os.path.join(LOG_DIR, f"feature_vector_{base_name}.png")
                visualize_feature_vector(features_for_plot, edges, faces, num_vertices, feature_vector_out)
            except Exception as e_feat:
                print(f"Warning: could not visualize feature vector for {base_name}: {e_feat}")

            # For inference, construct the same feature vector expected by the model.
            # We use the shared preprocessing function to ensure consistency with training data.
            if TTA_ENABLE and TTA_SAMPLES > 1:
                preds = []
                for _ in range(TTA_SAMPLES):
                    features, _, _ = augment_and_feature_engineer_sample_numpy(
                        initial_full_data=initial_full,
                        initial_target=initial_target,
                        flip_prob=0.0,  # No flipping for inference TTA (only positional noise)
                        noise_stddev=TTA_NOISE_STDDEV_PIXELS,
                        edges=edges,
                        faces=faces,
                        image_width=IMAGE_WIDTH,
                        image_height=IMAGE_HEIGHT,
                        num_vertices=num_vertices
                    )
                    X = features.astype(np.float32)[None, :]
                    preds.append(model.predict(X, verbose=0)[0])
                pred_norm = np.mean(np.stack(preds, axis=0), axis=0)
            else:
                # Single pass, no noise (or consistent baseline)
                features, _, _ = augment_and_feature_engineer_sample_numpy(
                    initial_full_data=initial_full,
                    initial_target=initial_target,
                    flip_prob=0.0,
                    noise_stddev=0.0,
                    edges=edges,
                    faces=faces,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                    num_vertices=num_vertices
                )
                X = features.astype(np.float32)[None, :]
                pred_norm = model.predict(X, verbose=0)[0]  # (num_vertices, 2)

            # Print pixel coordinates: visible -> GT, hidden -> Predicted
            try:
                img_dims = np.array([IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float32)
                gt_pixels = initial_target[:, 1:3] * img_dims
                pred_pixels = pred_norm * img_dims
                visible_mask = (initial_target[:, 3] == 1.0)
                print(f"\nSample {i+1:03d}: {base_name} — Pixel coordinates")
                print("Idx  Vis   X       Y")
                for vi in range(num_vertices):
                    if visible_mask[vi]:
                        x, y = gt_pixels[vi]
                        print(f"{vi:3d}   V   {x:7.1f} {y:7.1f}")
                    else:
                        x, y = pred_pixels[vi]
                        print(f"{vi:3d}   H   {x:7.1f} {y:7.1f}")
            except Exception as e_print:
                print(f"Warning: failed to print pixel coordinates for {base_name}: {e_print}")

            out_path = os.path.join(LOG_DIR, f"prediction_viz_{base_name}.png")
            visualize_prediction(image_path, initial_target, pred_norm, edges, out_path, IMAGE_WIDTH, IMAGE_HEIGHT)
            # Post-prediction full overlay with mixed vertices and face fills
            try:
                pred_overlay_out = os.path.join(LOG_DIR, f"prediction_overlay_{base_name}.png")
                visualize_prediction_full(image_path, initial_target, pred_norm, edges, faces, pred_overlay_out, IMAGE_WIDTH, IMAGE_HEIGHT)
            except Exception as e_pp:
                print(f"Warning: could not create post-prediction overlay for {base_name}: {e_pp}")
        except Exception as e:
            print(f"Skipping {jf} due to error: {e}")
            continue


if __name__ == "__main__":
    main()
