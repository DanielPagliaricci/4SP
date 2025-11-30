#!/usr/bin/env python3
# utils.py

import numpy as np
import json
from scipy.spatial import ConvexHull

def normalize_quaternion(q):
    """Return a unit quaternion from the input tuple/list q."""
    q = np.array(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        return tuple(q / norm)
    return tuple(q)

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) in radians to a quaternion (w, x, y, z)
    using standard formulas.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)

def quaternion_to_euler(q):
    """
    Convert a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in radians.
    """
    w, x, y, z = q
    # Roll (x-axis rotation)
    t0 = 2.0 * (w*x + y*z)
    t1 = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = 2.0 * (w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw

def polygon_area(coords):
    """Calculates the area of a polygon using the Shoelace formula."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def compute_hybrid_features(x, edges, faces, image_width=1920, image_height=1080):
    """
    Compute a hybrid feature vector from raw vertex data x (flattened N×3 array).
    Output dims depend on N, number of edges and faces provided.
    """
    num_vertices = int(len(x) // 3)
    v_pixels = x.reshape(num_vertices, 3).copy() # Keep a copy with pixel coordinates
    v_normalized = v_pixels.copy()

    # Normalize pixel coordinates for most features
    v_normalized[:, 1] = v_normalized[:, 1] / image_width
    v_normalized[:, 2] = v_normalized[:, 2] / image_height
    vertex_features = v_normalized.flatten()

    # Edges are now passed as an argument
    edge_features = []
    for (i, j) in edges:
        if i >= num_vertices or j >= num_vertices:
            # Edge references a vertex outside current sample; skip safely
            edge_features.extend([0.0, 0.0])
            continue
        if v_normalized[i, 0] == 1.0 and v_normalized[j, 0] == 1.0:
            dx = v_normalized[i, 1] - v_normalized[j, 1]
            dy = v_normalized[i, 2] - v_normalized[j, 2]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            norm_angle = (angle + np.pi) / (2*np.pi)
            edge_features.extend([distance, norm_angle])
        else:
            edge_features.extend([0.0, 0.0])
    edge_features = np.array(edge_features, dtype=np.float32)

    # Face Centroid and Area Features
    face_centroid_features = []
    face_area_features = []
    total_image_area = image_width * image_height

    if faces is not None:
        for face_indices in faces:
            # Guard against out-of-range indices in faces
            if any(idx >= num_vertices for idx in face_indices):
                face_centroid_features.extend([0.0, 0.0])
                face_area_features.append(0.0)
                continue
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

    # Convex Hull Features
    visible_vertices_pixels = v_pixels[v_pixels[:, 0] == 1.0][:, 1:3]
    
    hull_area, hull_perimeter, hull_num_vertices = 0.0, 0.0, 0.0

    if len(visible_vertices_pixels) >= 3:
        try:
            hull = ConvexHull(visible_vertices_pixels)
            hull_area = hull.volume
            hull_perimeter = hull.area
            hull_num_vertices = len(hull.vertices)
        except Exception:
            pass

    image_diagonal = np.sqrt(image_width**2 + image_height**2)
    normalized_hull_area = hull_area / (image_width * image_height)
    normalized_hull_perimeter = hull_perimeter / image_diagonal if image_diagonal > 0 else 0.0
    normalized_hull_num_vertices = hull_num_vertices / float(num_vertices if num_vertices > 0 else 1)

    hull_features = np.array([
        normalized_hull_area, 
        normalized_hull_perimeter, 
        normalized_hull_num_vertices
    ], dtype=np.float32)

    hybrid = np.concatenate([vertex_features, edge_features, face_centroid_features, face_area_features, hull_features])
    return hybrid

def compute_hybrid_features_all_vertices(x, edges, faces, image_width=1920, image_height=1080):
    """
    Compute a hybrid feature vector from raw vertex data x (flattened N×3 array),
    using ALL vertices (visible and non-visible) for edges, faces, and convex hull.

    Layout matches compute_hybrid_features:
    - First 3*N: vertex data (flag, norm_x, norm_y) for each vertex
    - Next 2*len(edges): per-edge features [distance, norm_angle] for every edge
    - Next 2*len(faces): face centroid (norm_x, norm_y) for every face
    - Next len(faces): normalized projected area per face
    - Final 3: convex hull features of ALL vertices: [normalized_area, normalized_perimeter, normalized_vertex_count]
    """
    num_vertices = int(len(x) // 3)
    v_pixels = x.reshape(num_vertices, 3).copy()
    v_normalized = v_pixels.copy()

    # Normalize pixel coordinates
    v_normalized[:, 1] = v_normalized[:, 1] / image_width
    v_normalized[:, 2] = v_normalized[:, 2] / image_height
    vertex_features = v_normalized.flatten()

    # Edge features: compute for all edges regardless of visibility, guard indices
    edge_features = []
    for (i, j) in edges:
        if i >= num_vertices or j >= num_vertices:
            # distance, sin(angle), cos(angle)
            edge_features.extend([0.0, 0.0, 0.0])
            continue
        dx = v_normalized[i, 1] - v_normalized[j, 1]
        dy = v_normalized[i, 2] - v_normalized[j, 2]
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        # Use sin/cos encoding for wrapped angle
        edge_features.extend([distance, np.sin(angle), np.cos(angle)])
    edge_features = np.array(edge_features, dtype=np.float32)

    # Face Centroid and Area Features: compute for all faces regardless of visibility, guard indices
    face_centroid_features = []
    face_area_features = []
    total_image_area = image_width * image_height

    if faces is not None:
        for face_indices in faces:
            if any(idx >= num_vertices for idx in face_indices):
                face_centroid_features.extend([0.0, 0.0])
                face_area_features.append(0.0)
                continue
            face_vertices_norm_coords = v_normalized[face_indices, 1:3]
            centroid = np.mean(face_vertices_norm_coords, axis=0)
            face_centroid_features.extend(centroid)

            face_vertices_pixel_coords = v_pixels[face_indices, 1:3]
            pixel_area = polygon_area(face_vertices_pixel_coords)
            normalized_area = pixel_area / total_image_area
            face_area_features.append(normalized_area)
    face_centroid_features = np.array(face_centroid_features, dtype=np.float32)
    face_area_features = np.array(face_area_features, dtype=np.float32)

    # Convex Hull Features: use ALL vertices
    all_vertices_pixels = v_pixels[:, 1:3]
    hull_area, hull_perimeter, hull_num_vertices = 0.0, 0.0, 0.0
    if num_vertices >= 3:
        try:
            hull = ConvexHull(all_vertices_pixels)
            hull_area = hull.volume
            hull_perimeter = hull.area
            hull_num_vertices = len(hull.vertices)
        except Exception:
            pass

    image_diagonal = np.sqrt(image_width**2 + image_height**2)
    normalized_hull_area = hull_area / (image_width * image_height)
    normalized_hull_perimeter = hull_perimeter / image_diagonal if image_diagonal > 0 else 0.0
    normalized_hull_num_vertices = hull_num_vertices / float(num_vertices if num_vertices > 0 else 1)

    hull_features = np.array([
        normalized_hull_area,
        normalized_hull_perimeter,
        normalized_hull_num_vertices
    ], dtype=np.float32)

    hybrid = np.concatenate([vertex_features, edge_features, face_centroid_features, face_area_features, hull_features])
    return hybrid

def load_vertex_data(json_path, num_vertices=None):
    """
    Load the raw vertex data from a JSON file.
    If num_vertices is None, infer size from the max vertex index present.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if num_vertices is None:
        max_idx = -1
        for vertex in data.get('visible_vertices', []):
            max_idx = max(max_idx, int(vertex['index']))
        for vertex in data.get('non_visible_vertices', []):
            max_idx = max(max_idx, int(vertex['index']))
        num_vertices = max_idx + 1 if max_idx >= 0 else 0

    result = np.zeros((num_vertices, 3), dtype=np.float32)
    for vertex in data.get('visible_vertices', []):
        idx = int(vertex['index'])
        if 0 <= idx < num_vertices:
            result[idx] = [1.0, vertex['pixel_coordinates'][0], vertex['pixel_coordinates'][1]]
    for vertex in data.get('non_visible_vertices', []):
        idx = int(vertex['index'])
        if 0 <= idx < num_vertices:
            result[idx] = [0.0, vertex['pixel_coordinates'][0], vertex['pixel_coordinates'][1]]
    return result.flatten()

def load_quaternion_from_json(json_path):
    """
    Extract the target rotation from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "rotation_quaternion" in data:
        quat = data["rotation_quaternion"]
        q = [quat["w"], quat["x"], quat["y"], quat["z"]]
    elif "quaternion" in data:
        quat = data["quaternion"]
        q = [quat["w"], quat["x"], quat["y"], quat["z"]]
    elif "rotation" in data:
        rot = data["rotation"]
        q = euler_to_quaternion(rot["x"], rot["y"], rot["z"])
    else:
        raise ValueError("Rotation data not found in JSON")
    q = normalize_quaternion(q)
    return np.array(q, dtype=np.float32)

def load_features_and_labels(json_path, num_vertices=None):
    """
    Load both raw vertex data and rotation quaternion from a single JSON file.
    This is more efficient as it reads the file only once.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Determine number of vertices if not provided
    if num_vertices is None:
        max_idx = -1
        for vertex in data.get('visible_vertices', []):
            max_idx = max(max_idx, int(vertex['index']))
        for vertex in data.get('non_visible_vertices', []):
            max_idx = max(max_idx, int(vertex['index']))
        num_vertices = max_idx + 1 if max_idx >= 0 else 0

    # Load vertex data
    result = np.zeros((num_vertices, 3), dtype=np.float32)
    for vertex in data.get('visible_vertices', []):
        idx = int(vertex['index'])
        if 0 <= idx < num_vertices:
            result[idx] = [1.0, vertex['pixel_coordinates'][0], vertex['pixel_coordinates'][1]]
    for vertex in data.get('non_visible_vertices', []):
        idx = int(vertex['index'])
        if 0 <= idx < num_vertices:
            result[idx] = [0.0, vertex['pixel_coordinates'][0], vertex['pixel_coordinates'][1]]
    vertex_data = result.flatten()

    # Load quaternion data
    if "rotation_quaternion" in data:
        quat = data["rotation_quaternion"]
        q = [quat["w"], quat["x"], quat["y"], quat["z"]]
    elif "quaternion" in data:
        quat = data["quaternion"]
        q = [quat["w"], quat["x"], quat["y"], quat["z"]]
    elif "rotation" in data:
        rot = data["rotation"]
        q = euler_to_quaternion(rot["x"], rot["y"], rot["z"])
    else:
        raise ValueError("Rotation data not found in JSON")
    
    quaternion = normalize_quaternion(q)
    quaternion_data = np.array(quaternion, dtype=np.float32)

    return vertex_data, quaternion_data 

def load_mesh_geometry(mesh_path):
    """
    Load edges, faces, and total vertex count from a mesh information JSON.
    Falls back to a default cube geometry if the file is missing or malformed.

    Returns: (edges, faces, total_vertices)
      - edges: list of (i, j)
      - faces: list of tuples of vertex indices (triangles/quads)
      - total_vertices: int
    """
    default_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    default_faces = [
        (0, 4, 6, 2), (3, 2, 6, 7), (7, 6, 4, 5),
        (5, 1, 3, 7), (1, 0, 2, 3), (5, 4, 0, 1),
    ]
    try:
        with open(mesh_path, "r") as f:
            mesh_info = json.load(f)
        edge_list = mesh_info.get("edge_composition")
        if edge_list:
            edges = [tuple(int(x) for x in pair) for pair in edge_list]
        else:
            # Derive undirected unique edges from vertex_connectivity
            connectivity = mesh_info.get("vertex_connectivity", {})
            unique_edges = set()
            for vertex_key, neighbors in connectivity.items():
                vertex_index = int(vertex_key)
                for neighbor in neighbors:
                    a, b = sorted((vertex_index, int(neighbor)))
                    unique_edges.add((a, b))
            edges = sorted(unique_edges)
        face_list = mesh_info.get("face_composition", [])
        faces = [tuple(int(x) for x in face_vertices) for face_vertices in face_list]
        total_vertices = int(mesh_info.get("total_vertices", 8))
        return edges, faces, total_vertices
    except Exception:
        # Fallback to default cube
        return default_edges, default_faces, 8