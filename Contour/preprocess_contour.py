#!/usr/bin/env python3
"""
Preprocess and visualize object contours from JSON annotations.

This script:
  - Loads JSON files describing an object's visible vertices (pixel coordinates)
  - Computes a 2D contour from the visible vertices (convex hull)
  - Finds the matching reference image by filename stem
  - Overlays the contour and visible points on the image
  - Displays up to NUM_SAMPLES_TO_SHOW examples

All configuration is hard-coded below. Do not use argparse per requirements.
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple, Dict, Set
import time

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.patches import Polygon as MplPolygon

# TODO: OUT OF SCOPE VERTICES ARE NOT BEEN CONSIDERED TO CALCULATE THE MASK, WE NEED TO ADD THEM TO THE MASK CALCULATION.
#############################
# CONFIGURATIONS #
#############################

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(BASE_DIR)

JSON_DIR: str = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "JSON_Data_MASK")
IMAGES_DIR: str = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Rendered_Images")
MESH_DIR: str = os.path.join(ROOT_DIR, "Database" , "Only_Obj_Env" , "Object_Information", "mesh_information_MASK_TANGO.json")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
print(BASE_DIR)
print(JSON_DIR)
print(IMAGES_DIR)
print(MESH_DIR)


NUM_SAMPLES_TO_SHOW: int = 3
MAX_PAIRED_FILES: Optional[int] = 40000

# Visualization settings
FIGSIZE_PER_SAMPLE: Tuple[float, float] = (4.0, 4.0)
CONTOUR_COLOR: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # red in matplotlib RGB
CONTOUR_LINEWIDTH: float = 2.0
POINT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 0.0)  # yellow
POINT_SIZE: float = 15.0

# Dataset save settings
SAVE_DATASET: bool = True
TARGET_SIZE: Tuple[int, int] = (256, 256)  # (width, height) for saved dataset

DATASET_OUT_DIR: str = os.path.dirname(os.path.abspath(__file__))  # where to save X_data.npy, Y_data.npy
X_DATA_PATH: str = os.path.join(DATASET_OUT_DIR, "X_data.npy")
Y_DATA_PATH: str = os.path.join(DATASET_OUT_DIR, "Y_data.npy")

# Mask construction settings
USE_NONVISIBLE_FOR_MASK: bool = False  # if True, include non-visible vertices for mask faces

# Shape extraction settings
USE_CONCAVE_HULL: bool = False  # revert to convex hull (older behavior)
ALPHA_SHAPE: float = 0.05       # bigger => smoother (closer to convex); smaller => more concave. Tune per dataset


# ==========================
# Core functionality
# ==========================

def list_json_files(json_dir: str) -> List[str]:
    """Return sorted list of JSON file paths in the given directory."""
    if not os.path.isdir(json_dir):
        print(f"[WARN] JSON directory not found: {json_dir}")
        return []
    files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    files.sort()
    return files


def load_json(json_path: str) -> Optional[dict]:
    """Load a single JSON file and return its content as dict, or None on error."""
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:  # noqa: BLE001 (explicit broad except acceptable for script robustness)
        print(f"[ERROR] Failed to load JSON '{json_path}': {exc}")
        return None


def extract_visible_pixel_points(annotation: dict) -> np.ndarray:
    """Extract Nx2 array of (x, y) pixel coordinates from 'visible_vertices'.

    Expected JSON structure (example):
      {
        "visible_vertices": [
           {"index": 0, "coordinates": [...], "pixel_coordinates": [x, y]},
           ...
        ],
        ...
      }
    """
    visible = annotation.get("visible_vertices", [])
    points: List[Tuple[float, float]] = []
    for v in visible:
        pix = v.get("pixel_coordinates")
        if (
            isinstance(pix, list)
            and len(pix) == 2
            and all(isinstance(val, (int, float)) for val in pix)
        ):
            # Ensure (x, y) order
            points.append((float(pix[0]), float(pix[1])))
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def extract_visible_vertex_map(annotation: dict) -> Dict[int, np.ndarray]:
    """Return mapping from vertex index -> (x,y) pixel coordinates for visible vertices."""
    visible = annotation.get("visible_vertices", [])
    vis_map: Dict[int, np.ndarray] = {}
    for v in visible:
        idx = v.get("index")
        pix = v.get("pixel_coordinates")
        if (
            isinstance(idx, int)
            and isinstance(pix, list)
            and len(pix) == 2
            and all(isinstance(val, (int, float)) for val in pix)
        ):
            vis_map[idx] = np.asarray([float(pix[0]), float(pix[1])], dtype=float)
    return vis_map


def extract_all_vertex_map(annotation: dict) -> Dict[int, np.ndarray]:
    """Return mapping from vertex index -> (x,y) for both visible and non-visible vertices if present.

    Tries common keys for non-visible vertices and merges them with visible ones (visible takes precedence).
    """
    result: Dict[int, np.ndarray] = {}
    # Start with non-visible candidates so that visible overwrites
    nonvisible_keys = [
        "non_visible_vertices",
        "invisible_vertices",
        "nonvisible_vertices",
        "hidden_vertices",
        "occluded_vertices",
        "all_vertices",
    ]
    for key in nonvisible_keys:
        items = annotation.get(key, [])
        if isinstance(items, list):
            for v in items:
                idx = v.get("index")
                pix = v.get("pixel_coordinates")
                if (
                    isinstance(idx, int)
                    and isinstance(pix, list)
                    and len(pix) == 2
                    and all(isinstance(val, (int, float)) for val in pix)
                ):
                    result[idx] = np.asarray([float(pix[0]), float(pix[1])], dtype=float)
    # Merge visible on top
    result.update(extract_visible_vertex_map(annotation))
    return result


def clamp_vertex_map_to_bounds(vertex_map: Dict[int, np.ndarray], width: int, height: int) -> Dict[int, np.ndarray]:
    """Clamp all coordinates to image bounds [0,width-1], [0,height-1]."""
    if not vertex_map:
        return vertex_map
    w_max = max(0, int(width) - 1)
    h_max = max(0, int(height) - 1)
    clamped: Dict[int, np.ndarray] = {}
    for k, v in vertex_map.items():
        x = float(v[0])
        y = float(v[1])
        x = 0.0 if x < 0.0 else x
        y = 0.0 if y < 0.0 else y
        x = float(w_max) if x > float(w_max) else x
        y = float(h_max) if y > float(h_max) else y
        clamped[k] = np.asarray([x, y], dtype=float)
    return clamped


def compute_visible_faces_pixels(face_composition: List[List[int]], visible_map: Dict[int, np.ndarray]) -> List[np.ndarray]:
    """Build list of visible face polygons (Nx2 arrays) using mesh faces and per-image visible vertices.

    A face is considered visible if all its vertices are present in visible_map.
    """
    faces_pixels: List[np.ndarray] = []
    for face in face_composition:
        try:
            pts = [visible_map[v_idx] for v_idx in face]
        except KeyError:
            continue
        if len(pts) >= 3:
            faces_pixels.append(np.vstack(pts))
    return faces_pixels


def find_matching_image(json_path: str, images_dir: str, extensions: Tuple[str, ...]) -> Optional[str]:
    """Find an image in 'images_dir' whose stem matches the base name of 'json_path'.

    For '.../name.json', we search for '.../name{ext}' where ext in extensions.
    Returns the first match (by extension order) or None if none found.
    """
    if not os.path.isdir(images_dir):
        print(f"[WARN] Images directory not found: {images_dir}")
        return None

    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in extensions:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def compute_convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """Compute the convex hull of 2D points using Andrew's monotonic chain.

    Args:
        points_xy: (N, 2) array of points as (x, y)

    Returns:
        (M, 2) array of hull vertices ordered counter-clockwise.
        If N < 3, returns the unique points.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] <= 1:
        return pts.copy()

    # Sort lexicographically by x, then y
    sorted_pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1])) if len(lower) + len(upper) > 2 else sorted_pts
    return hull


def _triangle_circumradius(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p0 - p2)
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    if area2 == 0:
        return float("inf")
    area = 0.5 * area2
    R = (a * b * c) / (4.0 * area)
    return float(R)


def compute_alpha_shape(points_xy: np.ndarray, alpha: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 4:
        return compute_convex_hull(pts)
    try:
        tri = Delaunay(pts)
    except Exception:
        return compute_convex_hull(pts)

    keep_edges: Dict[Tuple[int, int], int] = {}
    if alpha <= 0:
        return compute_convex_hull(pts)
    R_thresh = 1.0 / float(alpha)

    for simplex in tri.simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        p0, p1, p2 = pts[i], pts[j], pts[k]
        R = _triangle_circumradius(p0, p1, p2)
        if R < R_thresh:
            for (a, b) in [(i, j), (j, k), (k, i)]:
                if a > b:
                    a, b = b, a
                keep_edges[(a, b)] = keep_edges.get((a, b), 0) + 1

    boundary_edges = [(a, b) for (a, b), cnt in keep_edges.items() if cnt == 1]
    if len(boundary_edges) == 0:
        return compute_convex_hull(pts)

    adjacency: Dict[int, Set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    start = min(adjacency.keys(), key=lambda idx: (pts[idx, 0], pts[idx, 1]))
    polygon_indices: List[int] = [start]
    current = start
    prev = None
    for _ in range(len(boundary_edges) + 5):
        neighbors = adjacency.get(current, set())
        if not neighbors:
            break
        next_candidates = [n for n in neighbors if n != prev]
        if not next_candidates:
            break
        nxt = min(next_candidates, key=lambda idx: (pts[idx, 0], pts[idx, 1]))
        polygon_indices.append(nxt)
        prev, current = current, nxt
        if nxt == start:
            break

    if len(polygon_indices) >= 2 and polygon_indices[0] == polygon_indices[-1]:
        polygon_indices = polygon_indices[:-1]
    if len(polygon_indices) < 3:
        return compute_convex_hull(pts)
    return pts[np.array(polygon_indices, dtype=int)]


def compute_contour(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if USE_CONCAVE_HULL:
        try:
            return compute_alpha_shape(points_xy, ALPHA_SHAPE)
        except Exception:
            return compute_convex_hull(points_xy)
    return compute_convex_hull(points_xy)


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img)


def resize_image_and_points(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image to target_size (width, height) and scale points accordingly.

    Returns resized_image, scaled_points.
    """
    orig_h, orig_w = image_rgb.shape[0], image_rgb.shape[1]
    tgt_w, tgt_h = target_size
    scale_x = tgt_w / float(orig_w)
    scale_y = tgt_h / float(orig_h)

    img_resized = np.asarray(Image.fromarray(image_rgb).resize((tgt_w, tgt_h), resample=Image.BILINEAR))

    if points_xy.size == 0:
        return img_resized, points_xy

    scaled = points_xy.copy()
    scaled[:, 0] = scaled[:, 0] * scale_x
    scaled[:, 1] = scaled[:, 1] * scale_y
    return img_resized, scaled


def rasterize_polygon_to_mask(
    polygon_xy: np.ndarray,
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """Rasterize a polygon (Nx2) into a filled binary mask of size (H, W).

    If polygon has fewer than 3 points, draw small disks at points instead.
    """
    width, height = canvas_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if polygon_xy.shape[0] >= 3:
        xy = [tuple(map(float, pt)) for pt in polygon_xy]
        draw.polygon(xy, outline=1, fill=1)
    elif polygon_xy.shape[0] == 2:
        # Draw a small line thickness between two points
        p1 = tuple(map(float, polygon_xy[0]))
        p2 = tuple(map(float, polygon_xy[1]))
        draw.line([p1, p2], fill=1, width=3)
    elif polygon_xy.shape[0] == 1:
        # Draw a small disk around the point
        x, y = polygon_xy[0]
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=1)

    mask = np.asarray(mask_img, dtype=np.uint8)
    return mask


def rasterize_faces_union_to_mask(
    faces_pixels: List[np.ndarray],
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """Rasterize a list of face polygons into a single union binary mask.

    faces_pixels: list of (N_i, 2) arrays in pixel coordinates
    canvas_size: (width, height)
    """
    width, height = canvas_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in faces_pixels:
        if poly is None or poly.size == 0 or poly.shape[0] < 3:
            continue
        xy = [tuple(map(float, pt)) for pt in poly]
        draw.polygon(xy, outline=1, fill=1)
    mask = np.asarray(mask_img, dtype=np.uint8)
    return mask


def mask_to_boundary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Return a 1-pixel boundary mask from a filled binary mask {0,1}."""
    if binary_mask.size == 0:
        return binary_mask
    m = (binary_mask > 0).astype(np.uint8)
    h_b, w_b = m.shape
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    center = p[1:h_b+1, 1:w_b+1]
    up = p[0:h_b, 1:w_b+1]
    down = p[2:h_b+2, 1:w_b+1]
    left = p[1:h_b+1, 0:w_b]
    right = p[1:h_b+1, 2:w_b+2]
    boundary = (center == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    return boundary.astype(np.uint8)


def rasterize_polygon_outline_to_mask(
    polygon_xy: np.ndarray,
    canvas_size: Tuple[int, int],
    line_width: int = 1,
) -> np.ndarray:
    """Rasterize only the polygon outline into a binary mask."""
    width, height = canvas_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    if polygon_xy.shape[0] >= 2:
        pts = [tuple(map(float, pt)) for pt in polygon_xy]
        pts.append(tuple(map(float, polygon_xy[0])))
        draw.line(pts, fill=1, width=max(1, int(line_width)))
    elif polygon_xy.shape[0] == 1:
        x, y = polygon_xy[0]
        r = max(1, int(line_width))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
    return np.asarray(mask_img, dtype=np.uint8)


def extract_outer_contour_from_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Extract the outer contour from a filled binary mask {0,1}.

    Tries OpenCV if available; falls back to simple boundary-pixel extraction
    with a concave hull over boundary points when OpenCV is not installed.
    Returns (M,2) float array of (x,y) pixel coordinates.
    """
    if binary_mask.size == 0 or binary_mask.max() == 0:
        return np.zeros((0, 2), dtype=float)
    try:
        import cv2  # type: ignore
        mask_uint8 = (binary_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 2), dtype=float)
        # Choose the largest contour by area
        areas = [cv2.contourArea(cnt) for cnt in contours]
        best = contours[int(np.argmax(areas))]
        best = best[:, 0, :]  # shape (M,2)
        return best.astype(float)
    except Exception:
        # Fallback: compute boundary pixels and run concave hull over them
        m = (binary_mask > 0).astype(np.uint8)
        h, w = m.shape
        # Pad to simplify neighbor checks
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        center = p[1:h+1, 1:w+1]
        up = p[0:h, 1:w+1]
        down = p[2:h+2, 1:w+1]
        left = p[1:h+1, 0:w]
        right = p[1:h+1, 2:w+2]
        boundary = (center == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
        ys, xs = np.nonzero(boundary)
        if xs.size < 3:
            return np.vstack([xs, ys]).T.astype(float)
        pts = np.vstack([xs.astype(float), ys.astype(float)]).T
        if ALPHA_SHAPE > 0:
            try:
                return compute_alpha_shape(pts, ALPHA_SHAPE)
            except Exception:
                pass
        return compute_convex_hull(pts)


def plot_overlay(
    ax: plt.Axes,
    image_rgb: np.ndarray,
    visible_points: np.ndarray,
    contour_points: np.ndarray,
    title: str,
) -> None:
    ax.imshow(image_rgb)
    ax.set_title(title)
    ax.axis("off")

    # Plot all visible points
    if visible_points.size > 0:
        ax.scatter(
            visible_points[:, 0],
            visible_points[:, 1],
            s=POINT_SIZE,
            c=[POINT_COLOR],
            marker="o",
            linewidths=0.5,
            edgecolors="black",
        )

    # Plot the contour polygon (close the loop)
    # (Removed red contour plotting)


def build_dataset(paired_paths: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) dataset, resizing all images to TARGET_SIZE and rasterizing contour masks.

    X: (N, H, W, 3) uint8 images
    Y: (N, H, W) uint8 segmentation maps with values {0,1}
    """
    tgt_w, tgt_h = TARGET_SIZE
    images_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    # Load mesh faces once
    mesh = load_json(MESH_DIR)
    mesh_faces: List[List[int]] = mesh.get("face_composition", []) if isinstance(mesh, dict) else []

    for json_path, image_path in paired_paths:
        ann = load_json(json_path)
        if ann is None:
            print(f"[WARN] Skipping (bad JSON): {os.path.basename(json_path)}")
            continue

        # Visible vertices (map) and raw points (for fallback scatter)
        # Choose vertex source per flag
        if USE_NONVISIBLE_FOR_MASK:
            tmp_map = extract_all_vertex_map(ann)
        else:
            tmp_map = extract_visible_vertex_map(ann)
        # Clamp to image bounds (original image size before resizing)
        img_rgb = load_image_rgb(image_path)
        orig_h, orig_w = img_rgb.shape[0], img_rgb.shape[1]
        visible_map = clamp_vertex_map_to_bounds(tmp_map, orig_w, orig_h)
        visible_pts = np.vstack(list(visible_map.values())) if visible_map else np.zeros((0, 2))
        
        # Resize image and scale points
        
        tgt_w, tgt_h = TARGET_SIZE
        scale_x = tgt_w / float(orig_w)
        scale_y = tgt_h / float(orig_h)
        img_resized = np.asarray(Image.fromarray(img_rgb).resize((tgt_w, tgt_h), resample=Image.BILINEAR))

        # Scale visible map
        scaled_map: Dict[int, np.ndarray] = {
            k: np.asarray([v[0] * scale_x, v[1] * scale_y], dtype=float) for k, v in visible_map.items()
        }
        faces_pixels = compute_visible_faces_pixels(mesh_faces, scaled_map) if mesh_faces else []

        if faces_pixels:
            # Build union mask of faces and extract contour
            union_mask = rasterize_faces_union_to_mask(faces_pixels, (tgt_w, tgt_h))
            contour_pts = extract_outer_contour_from_mask(union_mask)
        else:
            pts_scaled = visible_pts.copy()
            if pts_scaled.size > 0:
                pts_scaled[:, 0] *= scale_x
                pts_scaled[:, 1] *= scale_y
            contour_pts = compute_contour(pts_scaled) if pts_scaled.shape[0] >= 1 else np.zeros((0, 2))

        # Build Y as filled region mask (consistent with earlier pipeline)
        if faces_pixels:
            mask = union_mask
        else:
            mask = rasterize_polygon_to_mask(contour_pts, (tgt_w, tgt_h))

        images_list.append(img_resized.astype(np.uint8))
        masks_list.append(mask.astype(np.uint8))

    if not images_list:
        return np.zeros((0, tgt_h, tgt_w, 3), dtype=np.uint8), np.zeros((0, tgt_h, tgt_w), dtype=np.uint8)

    X = np.stack(images_list, axis=0)
    Y = np.stack(masks_list, axis=0)
    return X, Y


def build_and_save_dataset_memmap(paired_paths: List[Tuple[str, str]], x_path: str, y_path: str) -> Tuple[int, Tuple[int, int]]:
    """Write dataset directly to .npy files using memmap to avoid high RAM usage.

    Returns total_count and (tgt_w, tgt_h) used for shapes printing.
    """
    tgt_w, tgt_h = TARGET_SIZE
    total = len(paired_paths)
    if total == 0:
        # Create empty files
        _ = open_memmap(x_path, mode="w+", dtype=np.uint8, shape=(0, tgt_h, tgt_w, 3))
        _ = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(0, tgt_h, tgt_w))
        return 0, (tgt_w, tgt_h)

    x_mm = open_memmap(x_path, mode="w+", dtype=np.uint8, shape=(total, tgt_h, tgt_w, 3))
    y_mm = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(total, tgt_h, tgt_w))

    start = time.time()
    update_every = max(1, total // 100)  # print every ~1%

    # Load mesh faces once
    mesh = load_json(MESH_DIR)
    mesh_faces: List[List[int]] = mesh.get("face_composition", []) if isinstance(mesh, dict) else []

    for idx, (json_path, image_path) in enumerate(paired_paths):
        try:
            ann = load_json(json_path)
            if ann is None:
                raise ValueError("Invalid JSON")

            # Choose vertex source per flag
            if USE_NONVISIBLE_FOR_MASK:
                tmp_map = extract_all_vertex_map(ann)
            else:
                tmp_map = extract_visible_vertex_map(ann)
            # Clamp to image bounds
            image_rgb = load_image_rgb(image_path)
            h, w = image_rgb.shape[0], image_rgb.shape[1]
            visible_map = clamp_vertex_map_to_bounds(tmp_map, w, h)
            visible_pts = np.vstack(list(visible_map.values())) if visible_map else np.zeros((0, 2))
            img_rgb = load_image_rgb(image_path)

            # Resize and scale
            orig_h, orig_w = img_rgb.shape[0], img_rgb.shape[1]
            scale_x = tgt_w / float(orig_w)
            scale_y = tgt_h / float(orig_h)
            img_resized = np.asarray(Image.fromarray(img_rgb).resize((tgt_w, tgt_h), resample=Image.BILINEAR))

            scaled_map: Dict[int, np.ndarray] = {
                k: np.asarray([v[0] * scale_x, v[1] * scale_y], dtype=float) for k, v in visible_map.items()
            }
            faces_pixels = compute_visible_faces_pixels(mesh_faces, scaled_map) if mesh_faces else []

            if faces_pixels:
                union_mask = rasterize_faces_union_to_mask(faces_pixels, (tgt_w, tgt_h))
                contour_pts = extract_outer_contour_from_mask(union_mask)
            else:
                pts_scaled = visible_pts.copy()
                if pts_scaled.size > 0:
                    pts_scaled[:, 0] *= scale_x
                    pts_scaled[:, 1] *= scale_y
                contour_pts = compute_contour(pts_scaled) if pts_scaled.shape[0] >= 1 else np.zeros((0, 2))

            if faces_pixels:
                mask = union_mask
            else:
                mask = rasterize_polygon_to_mask(contour_pts, (tgt_w, tgt_h))

            x_mm[idx] = img_resized.astype(np.uint8)
            y_mm[idx] = mask.astype(np.uint8)
        except Exception as exc:
            print(f"[WARN] Failed processing index {idx} ({os.path.basename(json_path)}): {exc}. Filling zeros.")
            x_mm[idx] = 0
            y_mm[idx] = 0

        if (idx + 1) % update_every == 0 or (idx + 1) == total:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = total - (idx + 1)
            eta_sec = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * (idx + 1) / total
            print(f"[INFO] Processed {idx + 1}/{total} ({pct:.1f}%). ETA: {eta_sec/60.0:.1f} min")

    # Ensure data is flushed
    del x_mm
    del y_mm

    return total, (tgt_w, tgt_h)


def main() -> None:
    json_files = list_json_files(JSON_DIR)
    if not json_files:
        print("[INFO] No JSON files found. Please check JSON_DIR.")
        return

    # Filter to those that have a matching image
    paired: List[Tuple[str, str]] = []
    for jp in json_files:
        img_path = find_matching_image(jp, IMAGES_DIR, IMAGE_EXTENSIONS)
        if img_path is None:
            print(f"[WARN] No matching image for JSON: {os.path.basename(jp)}")
            continue
        paired.append((jp, img_path))
        if MAX_PAIRED_FILES is not None and len(paired) >= MAX_PAIRED_FILES:
            print(f"[INFO] Reached cap: using first {MAX_PAIRED_FILES} matched pairs (out of potentially more).")
            break

    if not paired:
        print("[INFO] No JSON/Image pairs found. Check filenames and directories.")
        return

    # Select up to NUM_SAMPLES_TO_SHOW for quick visual check (no resizing)
    if NUM_SAMPLES_TO_SHOW > 0:
        samples = paired[: NUM_SAMPLES_TO_SHOW]
        num = len(samples)

        fig, axes = plt.subplots(
            3,
            num,
            figsize=(FIGSIZE_PER_SAMPLE[0] * num, FIGSIZE_PER_SAMPLE[1] * 3),
            constrained_layout=True,
        )
        if num == 1:
            axes = np.array(axes).reshape(3, 1)

        # Load mesh faces once
        mesh = load_json(MESH_DIR)
        mesh_faces: List[List[int]] = mesh.get("face_composition", []) if isinstance(mesh, dict) else []

        for col_idx, (json_path, image_path) in enumerate(samples):
            ann = load_json(json_path)
            if ann is None:
                axes[0, col_idx].axis("off")
                axes[1, col_idx].axis("off")
                axes[0, col_idx].set_title(f"Failed to load: {os.path.basename(json_path)}")
                continue

            if USE_NONVISIBLE_FOR_MASK:
                tmp_map = extract_all_vertex_map(ann)
            else:
                tmp_map = extract_visible_vertex_map(ann)
            # Load image to know bounds for clamping and for display
            image_rgb = load_image_rgb(image_path)
            h, w = image_rgb.shape[0], image_rgb.shape[1]
            visible_map = clamp_vertex_map_to_bounds(tmp_map, w, h)
            visible_pts = np.vstack(list(visible_map.values())) if visible_map else np.zeros((0, 2))
            if visible_pts.shape[0] == 0:
                print(f"[WARN] No visible vertices in: {os.path.basename(json_path)}")

            # Faces and combined contour + union mask (no resizing in preview)
            faces_pixels = compute_visible_faces_pixels(mesh_faces, visible_map) if mesh_faces else []
            if faces_pixels:
                union_mask = rasterize_faces_union_to_mask(faces_pixels, (w, h))
                contour_pts = extract_outer_contour_from_mask(union_mask)
            else:
                union_mask = np.zeros((h, w), dtype=np.uint8)
                contour_pts = compute_contour(visible_pts) if visible_pts.shape[0] >= 1 else np.zeros((0, 2))

            title = f"{os.path.basename(image_path)}\nvisible: {visible_pts.shape[0]} | contour: {contour_pts.shape[0]}"
            plot_overlay(axes[0, col_idx], image_rgb, visible_pts, contour_pts, title)

            # Overlay faces with translucent polygons
            if faces_pixels:
                for poly in faces_pixels:
                    try:
                        patch = MplPolygon(poly, closed=True, fill=False, edgecolor=(0, 1, 0), linewidth=1.0, alpha=0.8)
                        axes[0, col_idx].add_patch(patch)
                    except Exception:
                        continue

            # Then: draw mask boundary pixels (cyan) using same coordinate system
            if faces_pixels:
                try:
                    m = (union_mask > 0).astype(np.uint8)
                    h_b, w_b = m.shape
                    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
                    center = p[1:h_b+1, 1:w_b+1]
                    up = p[0:h_b, 1:w_b+1]
                    down = p[2:h_b+2, 1:w_b+1]
                    left = p[1:h_b+1, 0:w_b]
                    right = p[1:h_b+1, 2:w_b+2]
                    boundary = (center == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
                    ys_b, xs_b = np.nonzero(boundary)
                    if xs_b.size > 0:
                        axes[0, col_idx].scatter(xs_b, ys_b, s=0.8, c=[(0.0, 1.0, 1.0)], linewidths=0, zorder=4)
                except Exception:
                    pass

            # (Removed red contour plotting for exact overlay)

            # Middle row: show union mask (black/white)
            axes[1, col_idx].imshow(union_mask, cmap="gray", vmin=0, vmax=1)
            axes[1, col_idx].axis("off")
            axes[1, col_idx].set_title("Union mask (faces)")

            # Bottom row: original image + final dataset contour (edge mask)
            axes[2, col_idx].imshow(image_rgb)
            axes[2, col_idx].axis("off")
            axes[2, col_idx].set_title("Final dataset contour")
            try:
                if faces_pixels:
                    dataset_edge = mask_to_boundary_mask(union_mask)
                else:
                    dataset_edge = rasterize_polygon_outline_to_mask(contour_pts, (w, h), line_width=1)
                m = (dataset_edge > 0).astype(np.uint8)
                ys_e, xs_e = np.nonzero(m)
                if xs_e.size > 0:
                    axes[2, col_idx].scatter(xs_e, ys_e, s=0.8, c=[(1.0, 0.0, 1.0)], linewidths=0, zorder=4)
            except Exception:
                pass

        plt.tight_layout()
        plt.show()

    # Build and save dataset
    if SAVE_DATASET:
        print("[INFO] Building dataset with memmap (low RAM), this may take a while...")
        os.makedirs(DATASET_OUT_DIR, exist_ok=True)
        count, (tgt_w, tgt_h) = build_and_save_dataset_memmap(paired, X_DATA_PATH, Y_DATA_PATH)
        print(f"[INFO] Done. Saved X to: {X_DATA_PATH} | shape=({count}, {tgt_h}, {tgt_w}, 3)")
        print(f"[INFO] Done. Saved Y to: {Y_DATA_PATH} | shape=({count}, {tgt_h}, {tgt_w})")


if __name__ == "__main__":
    main()


