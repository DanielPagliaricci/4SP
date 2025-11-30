import os
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

# Use the exact same utils used during preprocessing/training
from utils import (
    compute_hybrid_features_all_vertices,
    load_vertex_data as utils_load_vertex_data,
    load_quaternion_from_json as utils_load_quaternion,
    quaternion_to_euler as utils_quaternion_to_euler,
    load_mesh_geometry,
)
from train_rotation import geodesic_loss, geodesic_deg, UnitNormalize

#############################
# Configuration
#############################
# Note: Update these paths if your data or model is located elsewhere.
BASE_PATH = "/Users/dani/Desktop/ICTati/Second_Attempt_AngularDataBase/TANGO"

# Use the same dataset structure as preprocessing/training
#VERTICES_FOLDER = os.path.join(BASE_PATH, "vertices_data_sample_TANGO_REF_sample_article")
#IMAGES_FOLDER = os.path.join(BASE_PATH, "renders_sample_TANGO_sample")

VERTICES_FOLDER = os.path.join("/Users/dani/PycharmProjects/ML_GPU_TEST/IC_TATI_2/TANGO/zEPnP_RANSEC/preprocessed_same_noise")
#VERTICES_FOLDER = os.path.join(BASE_PATH, "vertices_data_sample_TANGO_REF_sample1")
IMAGES_FOLDER = os.path.join(BASE_PATH, "renders_sample_TANGO_sample1")


MODEL_PATH = "models_good_250/best_model.keras"  # Relative path to the model saved by the training script
LOG_DIR = "testing_plots_enchanced_ref_comparison"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
NUM_SAMPLES_TO_VISUALIZE = 10
TRIM_WORST_FRACTION = 0.10  # Fraction of worst samples (by angular error) to exclude from summary metrics
MAX_ERROR_ALERT_DEG = 150.0  # Print samples with angular error >= this threshold

# Mesh information JSON — match preprocessing/training (REF mesh)
OBJECT_MESH_PATH = os.path.join(
    BASE_PATH,
    "Object_Information",
    "mesh_information_REF_TANGO.json",
)


def _load_mesh_geometry(mesh_path):
    """Deprecated local helper. Use utils.load_mesh_geometry instead."""
    return load_mesh_geometry(mesh_path)


EDGES, FACES, TOTAL_VERTICES = load_mesh_geometry(OBJECT_MESH_PATH)

#############################
# Helper Functions
#############################


def quaternion_to_euler(q):
    """Delegate to utils' implementation for consistency."""
    return utils_quaternion_to_euler(q)


def load_vertex_data(json_path):
    """Use the same loader as in preprocessing/training (utils) with fixed vertex count."""
    return utils_load_vertex_data(json_path, num_vertices=int(TOTAL_VERTICES))


def load_quaternion_from_json(json_path):
    """Use the same quaternion loader as in training/preprocessing (utils)."""
    return utils_load_quaternion(json_path)


def load_dataset(vertices_folder, edges, faces, image_width=1920, image_height=1080):
    """
    Loads the dataset from the folder of JSON files.
    For each file, computes a 69-dim hybrid feature vector and extracts the target quaternion (4-dim),
    matching the preprocessing/training pipeline.
    Returns:
       X: (N, 69) numpy array of hybrid features.
       Y: (N, 4) numpy array of quaternions.
       raw_X: (N, 24) numpy array of raw vertex data [vis, px, py].
       json_files: List of paths to the JSON files.
    """
    json_files = sorted(
        [os.path.join(vertices_folder, f) for f in os.listdir(vertices_folder) if f.endswith(".json")]
    )
    X_list = []
    Y_list = []
    raw_X_list = []
    for jf in json_files:
        try:
            raw_vertex = load_vertex_data(jf)
            hybrid = compute_hybrid_features_all_vertices(raw_vertex, edges, faces, image_width, image_height)
            q = load_quaternion_from_json(jf)
            X_list.append(hybrid)
            Y_list.append(q)
            raw_X_list.append(raw_vertex)
        except Exception as e:
            print(f"Skipping {jf} due to error: {e}")
            continue
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    raw_X = np.array(raw_X_list, dtype=np.float32)
    return X, Y, raw_X, json_files


#############################
# Prediction Analysis Function
#############################


def analyze_predictions(model, X, Y, json_files, log_dir=None, save_json=True, save_csv=True):
    """
    Predicts quaternions for all samples in X, converts both predictions and ground truth to Euler angles,
    and prints a comparison for each sample. Also computes and prints overall error statistics.
    """
    print("Analyzing individual predictions...")
    preds = model.predict(X)
    num_samples = X.shape[0]
    angular_errors_deg = []
    roll_errors_deg = []
    pitch_errors_deg = []
    yaw_errors_deg = []
    sample_records = []

    for i in range(num_samples):
        q_true = Y[i]
        q_pred = preds[i]

        # Normalize predicted quaternion for safety
        q_pred /= np.linalg.norm(q_pred)

        euler_true = quaternion_to_euler(q_true)
        euler_pred = quaternion_to_euler(q_pred)
        euler_true_deg = np.degrees(euler_true)
        euler_pred_deg = np.degrees(euler_pred)

        # Total Angular Error (from quaternion)
        dot = np.clip(np.abs(np.dot(q_true, q_pred)), -1.0, 1.0)
        angular_error_rad = 2 * np.arccos(dot)
        angular_error_deg = np.degrees(angular_error_rad)
        angular_errors_deg.append(angular_error_deg)

        # Per-axis Euler Angle Errors
        roll_err_rad = euler_true[0] - euler_pred[0]
        pitch_err_rad = euler_true[1] - euler_pred[1]
        yaw_err_rad = euler_true[2] - euler_pred[2]

        roll_err_rad = (roll_err_rad + np.pi) % (2 * np.pi) - np.pi
        pitch_err_rad = (pitch_err_rad + np.pi) % (2 * np.pi) - np.pi
        yaw_err_rad = (yaw_err_rad + np.pi) % (2 * np.pi) - np.pi

        roll_errors_deg.append(np.degrees(np.abs(roll_err_rad)))
        pitch_errors_deg.append(np.degrees(np.abs(pitch_err_rad)))
        yaw_errors_deg.append(np.degrees(np.abs(yaw_err_rad)))

        print(f"Sample {i} (File: {os.path.basename(json_files[i])}):")
        print(f"  Ground truth Euler angles (deg): {euler_true_deg}")
        print(f"  Predicted Euler angles (deg):    {euler_pred_deg}")
        print(f"  Angular error: {angular_error_deg:.2f}°")
        print("-" * 50)

        sample_records.append({
            "file": os.path.basename(json_files[i]),
            "angular_error_deg": float(angular_error_deg),
            "roll_error_deg": float(roll_errors_deg[-1]),
            "pitch_error_deg": float(pitch_errors_deg[-1]),
            "yaw_error_deg": float(yaw_errors_deg[-1]),
            "gt_roll_deg": float(euler_true_deg[0]),
            "gt_pitch_deg": float(euler_true_deg[1]),
            "gt_yaw_deg": float(euler_true_deg[2]),
            "pred_roll_deg": float(euler_pred_deg[0]),
            "pred_pitch_deg": float(euler_pred_deg[1]),
            "pred_yaw_deg": float(euler_pred_deg[2]),
        })

    # Overall Analysis
    if angular_errors_deg:
        errors = np.array(angular_errors_deg)
        roll_errors_arr = np.array(roll_errors_deg)
        pitch_errors_arr = np.array(pitch_errors_deg)
        yaw_errors_arr = np.array(yaw_errors_deg)

        # Determine which samples to exclude (pass 1: identify outliers)
        num_exclude_frac = int(np.floor(TRIM_WORST_FRACTION * num_samples))
        num_exclude_frac = max(0, min(num_exclude_frac, max(0, num_samples - 1)))
        sorted_idx_desc = np.argsort(-errors)
        excluded_fraction = set(sorted_idx_desc[:num_exclude_frac])
        excluded_threshold = set(np.where(errors >= MAX_ERROR_ALERT_DEG)[0])
        excluded_idx = excluded_fraction.union(excluded_threshold)
        included_mask = np.ones(num_samples, dtype=bool)
        if len(excluded_idx) > 0:
            included_mask[list(excluded_idx)] = False

        errors_included = errors[included_mask]
        roll_included = roll_errors_arr[included_mask]
        pitch_included = pitch_errors_arr[included_mask]
        yaw_included = yaw_errors_arr[included_mask]

        # Annotate sample records with exclusion flag
        for i in range(num_samples):
            sample_records[i]["excluded_from_summary"] = bool(i in excluded_idx)

        # Print summary metrics (trimmed)
        print("\n" + "=" * 20 + " Pass 1: Outlier detection " + "=" * 20)
        print(f"Total samples analyzed: {num_samples}")
        print(f"Excluded {len(excluded_idx)} samples (worst {num_exclude_frac} by fraction + {len(excluded_threshold)} above {MAX_ERROR_ALERT_DEG:.1f}°)")
        if num_exclude_frac > 0:
            threshold_deg = float(errors[sorted_idx_desc[num_exclude_frac - 1]])
            print(f"Fraction exclusion cutoff (>=): {threshold_deg:.2f}°")

        print("\n" + "=" * 20 + " Pass 2: Metrics excluding outliers " + "=" * 20)
        print("\n--- Holistic Rotational Deviation (excluding outliers) ---")
        print(f"Mean Orientation Error (E_R): {np.mean(errors_included):.2f}°")

        print(f"\nOverall Angular Error Statistics (degrees, excluding outliers):")
        print(f"  Median: {np.median(errors_included):.2f}°")
        print(f"  Std Dev:  {np.std(errors_included):.2f}°")
        print(f"  Min:    {np.min(errors_included):.2f}°")
        print(f"  Max:    {np.max(errors_included):.2f}°")

        for axis_name, errors_axis in [
            ("Roll (X)", roll_included),
            ("Pitch (Y)", pitch_included),
            ("Yaw (Z)", yaw_included),
        ]:
            print(f"\nAbsolute Error for {axis_name} (degrees, excluding outliers):")
            print(f"  Mean:   {np.mean(errors_axis):.2f}°")
            print(f"  Median: {np.median(errors_axis):.2f}°")
            print(f"  Std Dev:  {np.std(errors_axis):.2f}°")
            print(f"  Min:    {np.min(errors_axis):.2f}°")
            print(f"  Max:    {np.max(errors_axis):.2f}°")
        # Identify which file corresponds to the max angular error in the trimmed set
        if errors_included.size > 0:
            included_indices = np.nonzero(included_mask)[0]
            max_included_subidx = int(np.argmax(errors_included))
            max_included_idx = int(included_indices[max_included_subidx])
            print(f"\nMax angular error sample (trimmed): idx={max_included_idx}, file={os.path.basename(json_files[max_included_idx])}, error={errors[max_included_idx]:.2f}°")

        # Print filenames of all excluded samples (outliers)
        if len(excluded_idx) > 0:
            excluded_files = [os.path.basename(json_files[int(i)]) for i in sorted(excluded_idx)]
            print("\nOutlier sample filenames (excluded):")
            for name in excluded_files:
                print(f"  {name}")
            # Save outliers list
            if log_dir is not None:
                try:
                    with open(os.path.join(log_dir, "outliers.txt"), "w") as f:
                        for name in excluded_files:
                            f.write(name + "\n")
                    with open(os.path.join(log_dir, "outliers.json"), "w") as f:
                        json.dump({
                            "indices": [int(i) for i in sorted(excluded_idx)],
                            "files": excluded_files,
                            "num_excluded": len(excluded_idx),
                            "num_excluded_by_fraction": len(excluded_fraction),
                            "num_excluded_by_threshold": len(excluded_threshold),
                            "threshold_deg": float(MAX_ERROR_ALERT_DEG),
                        }, f, indent=2)
                    print(f"Saved outlier lists to {os.path.join(log_dir, 'outliers.*')}")
                except Exception as e:
                    print(f"Failed to save outlier lists: {e}")

        # Print all samples with very large angular error (>= threshold)
        alert_indices = np.where(errors >= MAX_ERROR_ALERT_DEG)[0]
        if alert_indices.size > 0:
            print(f"\nSamples with angular error ≥ {MAX_ERROR_ALERT_DEG:.2f}°:")
            for idx in alert_indices:
                print(f"  idx={int(idx)}, file={os.path.basename(json_files[int(idx)])}, error={errors[int(idx)]:.2f}°, excluded_from_summary={bool(idx in excluded_idx)}")
            # Print filenames only for the alert set
            alert_files = [os.path.basename(json_files[int(idx)]) for idx in alert_indices]
            print("\nAlert sample filenames:")
            for name in alert_files:
                print(f"  {name}")
        print("=" * 58)

        # Save metrics
        if log_dir is not None:
            summary = {
                "total_samples": int(num_samples),
                "trim": {
                    "requested_fraction": float(TRIM_WORST_FRACTION),
                    "excluded_count": int(len(excluded_idx)),
                    "excluded_by_fraction": int(len(excluded_fraction)),
                    "excluded_by_threshold": int(len(excluded_threshold)),
                    "fraction_cutoff_deg": float(errors[sorted_idx_desc[num_exclude_frac - 1]]) if num_exclude_frac > 0 else None,
                    "threshold_deg": float(MAX_ERROR_ALERT_DEG),
                },
                "overall": {
                    "mean_deg": float(np.mean(errors_included)),
                    "median_deg": float(np.median(errors_included)),
                    "std_deg": float(np.std(errors_included)),
                    "min_deg": float(np.min(errors_included)),
                    "max_deg": float(np.max(errors_included)),
                },
                "per_axis": {
                    "roll": {
                        "mean_deg": float(np.mean(roll_included)),
                        "median_deg": float(np.median(roll_included)),
                        "std_deg": float(np.std(roll_included)),
                        "min_deg": float(np.min(roll_included)),
                        "max_deg": float(np.max(roll_included)),
                    },
                    "pitch": {
                        "mean_deg": float(np.mean(pitch_included)),
                        "median_deg": float(np.median(pitch_included)),
                        "std_deg": float(np.std(pitch_included)),
                        "min_deg": float(np.min(pitch_included)),
                        "max_deg": float(np.max(pitch_included)),
                    },
                    "yaw": {
                        "mean_deg": float(np.mean(yaw_included)),
                        "median_deg": float(np.median(yaw_included)),
                        "std_deg": float(np.std(yaw_included)),
                        "min_deg": float(np.min(yaw_included)),
                        "max_deg": float(np.max(yaw_included)),
                    },
                },
            }

            if save_json:
                try:
                    with open(os.path.join(log_dir, "metrics_summary.json"), "w") as f:
                        json.dump(summary, f, indent=2)
                    print(f"Saved metrics summary to {os.path.join(log_dir, 'metrics_summary.json')}")
                except Exception as e:
                    print(f"Failed to write metrics summary: {e}")

            if save_csv:
                try:
                    csv_path = os.path.join(log_dir, "per_sample_metrics.csv")
                    with open(csv_path, "w", newline="") as f:
                        fieldnames = [
                            "file",
                            "angular_error_deg",
                            "roll_error_deg",
                            "pitch_error_deg",
                            "yaw_error_deg",
                            "gt_roll_deg",
                            "gt_pitch_deg",
                            "gt_yaw_deg",
                            "pred_roll_deg",
                            "pred_pitch_deg",
                            "pred_yaw_deg",
                            "excluded_from_summary",
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(sample_records)
                    print(f"Saved per-sample metrics to {csv_path}")
                except Exception as e:
                    print(f"Failed to write per-sample metrics CSV: {e}")


#############################
# Visualization Function
#############################


def visualize_prediction(image_path, raw_vertex_data, q_true, q_pred, output_path, image_width=1920, image_height=1080):
    """
    Visualizes the ground truth vertex locations and compares rotation predictions.
    """
    try:
        img = plt.imread(image_path)
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Creating a blank placeholder.")
        img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 220

    fig, ax = plt.subplots(1, figsize=(12, 12 * (img.shape[0] / img.shape[1]) if img.shape[1] > 0 else 12))
    ax.imshow(img)

    num_vertices = int(len(raw_vertex_data) // 3)
    vertices = raw_vertex_data.reshape(num_vertices, 3)  # [vis, x, y]

    legend_handles = {}
    for i in range(num_vertices):
        is_visible = (vertices[i, 0] == 1.0)
        px, py = vertices[i, 1], vertices[i, 2]

        color = 'seagreen' if is_visible else 'khaki'
        label = 'Visible Vertex' if is_visible else 'Non-Visible Vertex'

        scatter = ax.scatter(px, py, c=color, s=120, edgecolors='black', zorder=3, alpha=0.9)
        if label not in legend_handles:
            legend_handles[label] = scatter

        ax.text(px + 15, py, str(i), color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, pad=0.2, edgecolor='none', boxstyle='round,pad=0.3'), zorder=4)

    # Optionally draw edges
    line = None
    for u, v_idx in EDGES:
        start_pos = vertices[u, 1:3]
        end_pos = vertices[v_idx, 1:3]
        # line, = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
        #                 color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    if line:
        legend_handles["Edges"] = line

    # Add rotation text
    euler_true_deg = np.degrees(quaternion_to_euler(q_true))
    euler_pred_deg = np.degrees(quaternion_to_euler(q_pred))

    text_str = (f"Ground Truth Euler (R, P, Y):\n"
                f"({euler_true_deg[0]:.1f}°, {euler_true_deg[1]:.1f}°, {euler_true_deg[2]:.1f}°)\n\n"
                f"Predicted Euler (R, P, Y):\n"
                f"({euler_pred_deg[0]:.1f}°, {euler_pred_deg[1]:.1f}°, {euler_pred_deg[2]:.1f}°)")

    ax.text(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        linespacing=1.4,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', edgecolor='black', linewidth=1.5, alpha=0.85),
    )

    ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title="Key", loc="upper right")
    ax.set_title(f"Rotation Prediction for {os.path.basename(image_path)}", fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")
    plt.close(fig)


#############################
# Main Script
#############################


def main():
    # Path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    log_dir = os.path.join(script_dir, LOG_DIR)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created log directory: {log_dir}")

    # Data Loading (train-compatible)
    print("Loading and processing dataset (train-compatible REF mesh)...")
    X, Y, raw_X, json_files = load_dataset(VERTICES_FOLDER, EDGES, FACES, IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"Loaded dataset: X shape: {X.shape}, Y shape: {Y.shape}")
    if X.shape[0] == 0:
        print("No data loaded. Please check your dataset folder and JSON format.")
        return

    # Model Loading
    custom_objects = {"geodesic_loss": geodesic_loss, "geodesic_deg": geodesic_deg, "UnitNormalize": UnitNormalize}
    try:
        try:
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
        model.summary()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Input dimension check (prevents shape mismatch)
    expected_input_dim = model.input_shape[-1]
    actual_input_dim = X.shape[1]
    if actual_input_dim != expected_input_dim:
        print(
            f"Feature dimension mismatch: X has {actual_input_dim} features but model expects {expected_input_dim}.\n"
            f"Check OBJECT_MESH_PATH and dataset (currently REF mesh at {OBJECT_MESH_PATH}) are aligned with training."
        )
        return

    # Analysis
    print("\nComparing predictions for all samples in the dataset:")
    analyze_predictions(model, X, Y, json_files, log_dir=log_dir, save_json=True, save_csv=True)

    # Visualization
    if NUM_SAMPLES_TO_VISUALIZE > 0:
        print(f"\nVisualizing predictions for the first {NUM_SAMPLES_TO_VISUALIZE} samples...")
        num_to_viz = min(NUM_SAMPLES_TO_VISUALIZE, X.shape[0])
        if num_to_viz > 0:
            preds = model.predict(X[:num_to_viz])
            for i in range(num_to_viz):
                base_name = os.path.splitext(os.path.basename(json_files[i]))[0]
                image_path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    potential_path = os.path.join(IMAGES_FOLDER, base_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                if not image_path:
                    continue
                output_viz_path = os.path.join(log_dir, f"prediction_viz_{base_name}.png")
                visualize_prediction(
                    image_path=image_path,
                    raw_vertex_data=raw_X[i],
                    q_true=Y[i],
                    q_pred=preds[i],
                    output_path=output_viz_path,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )


if __name__ == "__main__":
    main()


