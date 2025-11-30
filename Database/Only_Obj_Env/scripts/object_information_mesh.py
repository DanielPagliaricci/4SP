import bpy
import json
import os



# --- CONFIGURATION --- #
OBJECT_NAME = "Cube" # Name of the object to get the mesh information from in blender
BASE_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Object_Information")
OUTPUT_FILENAME = "mesh_information.json"

def get_mesh_information(object_name="Cube"):
    """
    Extracts vertex connectivity and face composition from a mesh object.

    Args:
        object_name (str): The name of the object to inspect.

    Returns:
        dict: A dictionary containing mesh information, or None if the object is not found or not a mesh.
    """
    # Find the object in the scene
    obj = bpy.data.objects.get(object_name)

    if not obj:
        print(f"Error: Object '{object_name}' not found.")
        return None

    if obj.type != 'MESH':
        print(f"Error: Object '{object_name}' is not a mesh.")
        return None

    mesh = obj.data

    # 1. Vertex-to-Vertex Links (Vertex Connectivity)
    # Initialize a dictionary for each vertex
    vertex_links = {i: [] for i in range(len(mesh.vertices))}

    # Iterate over each edge to find connected vertices
    for edge in mesh.edges:
        v1_idx, v2_idx = edge.vertices
        # Add a two-way connection
        if v2_idx not in vertex_links[v1_idx]:
            vertex_links[v1_idx].append(v2_idx)
        if v1_idx not in vertex_links[v2_idx]:
            vertex_links[v2_idx].append(v1_idx)
    
    # Sort the lists for consistent output
    for i in vertex_links:
        vertex_links[i].sort()

    # 2. Face Composition (Vertices per Face)
    face_composition = []
    for face in mesh.polygons:
        face_composition.append(list(face.vertices))

    # 3. Edge Composition (Vertices per Edge)
    edge_composition = []
    for edge in mesh.edges:
        edge_composition.append(list(edge.vertices))

    mesh_info = {
        "object_name": obj.name,
        "total_vertices": len(mesh.vertices),
        "total_edges": len(mesh.edges),
        "total_faces": len(mesh.polygons),
        "vertex_connectivity": vertex_links,
        "face_composition": face_composition,
        "edge_composition": edge_composition
    }

    return mesh_info

def save_info_to_json(data, output_path):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        output_path (str): The full path for the output JSON file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved mesh information to: {output_path}")
    except IOError as e:
        print(f"Error writing to file {output_path}: {e}")



def main():    
    # Ensure the output directory exists
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)
        
    output_filepath = os.path.join(BASE_OUTPUT_PATH, OUTPUT_FILENAME)

    # Get mesh info
    mesh_data = get_mesh_information(OBJECT_NAME)

    # Save to file
    if mesh_data:
        save_info_to_json(mesh_data, output_filepath)


main()
