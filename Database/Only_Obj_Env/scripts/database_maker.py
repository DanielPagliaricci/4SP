import bpy
import math
import mathutils
import json
import os
import numpy as np
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
import time

##################
# CONFIGURATIONS #
##################
np.random.seed(3)
TOTAL_ROTATIONS = 11 # Total number of random rotations to generate

BASE_OUTPUT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_SUBDIR = "Rendered_Images"
JSON_SUBDIR_REF = "JSON_Data_REF"
JSON_SUBDIR_MASK = "JSON_Data_MASK"
MESH_INFORMATION_DIR = os.path.join(BASE_OUTPUT_PATH, "Object_Information")

EPSILON = 1e-5
REF_OBJECT_NAME = "REF_TANGO"
MASK_OBJECT_NAME = "MASK_TANGO"
CAMERA_TYPE_FILTER = 'CAMERA'
LIGHT_OBJECT_NAME = "Sun" 


LIGHT_BOUNDARIES = [[11, 15], # X
                    [11, 15], # Y
                    [11, 15]] # Z


RANDOMIZE_OBJECT_POSITION = True
OBJECT_BOUNDARIES = [[-0.1, 0.1], # X
                     [-0.1, 0.1], # Y
                     [-0.1, 0.1]] # Z

DEBUG_MATERIALS = False
RANDOMIZE_LIGHT_POSITION = False

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

try:
    os.makedirs(MESH_INFORMATION_DIR, exist_ok=True)
    for _obj_name in [REF_OBJECT_NAME, MASK_OBJECT_NAME]:
        _mesh_info = get_mesh_information(_obj_name)
        if _mesh_info:
            _output_path = os.path.join(MESH_INFORMATION_DIR, f"mesh_information_{_obj_name}.json")
            save_info_to_json(_mesh_info, _output_path)
            print(f"Mesh information saved to: {_output_path}")
except Exception as e:
    print(f"Error saving mesh information: {e}")





# Utility functions _____________________________
def get_camera_direction(camera_obj):
    rot_matrix = camera_obj.matrix_world.to_3x3()
    forward = rot_matrix @ mathutils.Vector((0.0, 0.0, -1.0))
    forward.normalize()
    return forward

def get_face_normals_in_world(mesh_obj):
    mesh = mesh_obj.data
    mesh.calc_normals()
    normals_world = []
    
    for face in mesh.polygons:
        normal = mesh_obj.matrix_world.to_3x3() @ face.normal
        normal.normalize()
        normals_world.append(normal)
    return normals_world

def create_material(name, color):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
    return mat

class RotationManager:
    def __init__(self, total_rotations):
        self.total_rotations = total_rotations
        self.step_count = 0
        # Generate all rotations at initialization
        self.rotations = np.random.uniform(0, 360, (total_rotations, 3))

    def get_next_rotation(self):
        """
        Returns the next rotation tuple (x, y, z) in degrees
        Returns None if all rotations are completed
        """
        if self.step_count >= self.total_rotations:
            return None

        current_rotation = tuple(self.rotations[self.step_count])
        self.step_count += 1
        return current_rotation

    def get_progress(self):
        """Returns progress as a percentage"""
        return (self.step_count / self.total_rotations) * 100

    def get_rotation_index(self):
        """Returns current rotation index"""
        return self.step_count - 1

class VerticesDatabaseCreator(bpy.types.Operator):
    bl_idname = "render.vertices_database"
    bl_label = "Create Vertices Database"
    
    # Class properties
    OUTPUT_PATH = "//output" # This was likely a leftover or default, the actual paths are built in setup_output_directories
    timer_event = None # Internal state variables
    rendering = False # Internal state variables
    cancel_render = False # Internal state variables
    # Removed hardcoded variables: debug_materials, epsilon, render_images, total_rotations
    
    # Configuration (now using global variables)
    # Removed hardcoded variables from here
    
    def setup_output_directories(self):
        # Use global base path and subdir names
        self.base_path = Path(BASE_OUTPUT_PATH)
        self.image_path = self.base_path / IMAGE_SUBDIR
        self.json_path_ref = self.base_path / JSON_SUBDIR_REF
        self.json_path_mask = self.base_path / JSON_SUBDIR_MASK
        
        for path in [self.image_path, self.json_path_ref, self.json_path_mask]:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")
    
    def setup_materials(self, obj):
        # Use global DEBUG_MATERIALS flag
        if DEBUG_MATERIALS:
            visible_material = create_material("Visible_Material", (0.0, 1.0, 0.0, 1))
            nonvisible_material = create_material("Nonvisible_Material", (1.0, 0.0, 0.0, 1))
            
            if not obj.data.materials:
                obj.data.materials.append(nonvisible_material)
            if visible_material.name not in [mat.name for mat in obj.data.materials]:
                obj.data.materials.append(visible_material)
            if nonvisible_material.name not in [mat.name for mat in obj.data.materials]:
                obj.data.materials.append(nonvisible_material)
    
    def change_rotation(self, obj, rotation_angles):
        """Apply rotation to object (angles in degrees)"""
        x_rot = math.radians(rotation_angles[0])
        y_rot = math.radians(rotation_angles[1])
        z_rot = math.radians(rotation_angles[2])
        obj.rotation_euler = (x_rot, y_rot, z_rot)
        return x_rot, y_rot, z_rot
    
    def process_vertices(self, obj, camera, scene):
        mesh = obj.data
        resolution_x = scene.render.resolution_x
        resolution_y = scene.render.resolution_y
        
        # Get visible vertices
        visible_vertices_set, _, _ = self.process_face_visibility(obj, camera, scene)
        
        # Process all vertices
        q = obj.matrix_world.to_quaternion()
        vertices_data = {
            'visible_vertices': [],
            'non_visible_vertices': [],
            'rotation': {
                'x': obj.rotation_euler.x,
                'y': obj.rotation_euler.y,
                'z': obj.rotation_euler.z
            },
            'rotation_quaternion': {
                'w': q.w,
                'x': q.x,
                'y': q.y,
                'z': q.z
            }
        }
        
        for vertex_index in range(len(mesh.vertices)):
            vertex_info = self.process_single_vertex(
                vertex_index, mesh, obj, camera, scene,
                resolution_x, resolution_y
            )
            
            if vertex_index in visible_vertices_set and vertex_info['is_visible']:
                vertices_data['visible_vertices'].append(vertex_info['data'])
            else:
                vertices_data['non_visible_vertices'].append(vertex_info['data'])
        
        return vertices_data
    
    def process_face_visibility(self, obj, camera, scene):
        mesh = obj.data
        mesh.calc_normals()
        face_normals = get_face_normals_in_world(obj)
        camera_pos = camera.location
        
        visible_faces_info = []
        visible_vertices_set = set()
        
        for i, face in enumerate(mesh.polygons):
            face_center = obj.matrix_world @ face.center
            view_vector = (camera_pos - face_center).normalized()
            normal = face_normals[i]
            
            dot = normal.dot(view_vector)
            dot = max(min(dot, 1.0), -1.0)
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)

            if angle_deg < 90.0 + EPSILON:
                visibility = "Visible"
                visible_vertices_set.update(face.vertices)
                if DEBUG_MATERIALS:
                    material_index = obj.data.materials.find("Visible_Material")
                    face.material_index = material_index
            else:
                visibility = "Nonvisible"
                if DEBUG_MATERIALS:
                    material_index = obj.data.materials.find("Nonvisible_Material")
                    face.material_index = material_index

        return visible_vertices_set, visible_faces_info, []
    
    def process_single_vertex(self, vertex_index, mesh, obj, camera, scene, resolution_x, resolution_y):
        vertex = mesh.vertices[vertex_index]
        vertex_world_co = obj.matrix_world @ vertex.co
        co_ndc = world_to_camera_view(scene, camera, vertex_world_co)
        
        if co_ndc.z < 0:
            return {
                'is_visible': False,
                'data': {
                    'index': vertex_index,
                    'coordinates': [vertex_world_co.x, vertex_world_co.y, vertex_world_co.z],
                    'status': "Behind camera"
                }
            }
        
        # Compute subpixel coordinates in pixel space without clamping and without rounding
        pixel_x_f = co_ndc.x * resolution_x
        pixel_y_f = (1 - co_ndc.y) * resolution_y
        
        if 0 <= co_ndc.x <= 1 and 0 <= co_ndc.y <= 1:
            return {
                'is_visible': True,
                'data': {
                    'index': vertex_index,
                    'coordinates': [vertex_world_co.x, vertex_world_co.y, vertex_world_co.z],
                    'pixel_coordinates': [pixel_x_f, pixel_y_f]
                }
            }
        
        return {
            'is_visible': False,
            'data': {
                'index': vertex_index,
                'coordinates': [vertex_world_co.x, vertex_world_co.y, vertex_world_co.z],
                'status': "Outside view",
                'pixel_coordinates': [pixel_x_f, pixel_y_f]
            }
        }
    
    def make_filename(self, rotation_index, camera_name):
        """Creates standardized filename for both image and JSON files"""
        return f"rotation_{rotation_index}"
    
    def render_init(self, scene, depsgraph):
        self.rendering = True
        print("RENDER INIT")
    
    def render_complete(self, scene, depsgraph):
        self.rendering = False
        print("RENDER COMPLETE")
        self.process_next_item(bpy.context)
    
    def render_cancel(self, scene, depsgraph):
        self.cancel_render = True
        print("RENDER CANCELLED")
        self.process_next_item(bpy.context)
    
    def find_last_processed_rotation(self):
        """Find the last successfully processed rotation"""
        last_rotation = -1
        
        try:
            pattern_img = "rotation_*.png"
            pattern_json = "rotation_*.json"
            img_rotations = set()
            ref_rotations = set()
            mask_rotations = set()
            for file in self.image_path.glob(pattern_img):
                try:
                    # Handle filenames like rotation_123.png
                    parts = file.stem.split('_')
                    if len(parts) >= 2 and parts[0] == "rotation":
                         img_rotations.add(int(parts[1]))
                except:
                    pass
            for file in self.json_path_ref.glob(pattern_json):
                try:
                    parts = file.stem.split('_')
                    if len(parts) >= 2 and parts[0] == "rotation":
                        ref_rotations.add(int(parts[1]))
                except:
                    pass
            for file in self.json_path_mask.glob(pattern_json):
                try:
                    parts = file.stem.split('_')
                    if len(parts) >= 2 and parts[0] == "rotation":
                        mask_rotations.add(int(parts[1]))
                except:
                    pass
            completed_rotations = img_rotations.intersection(ref_rotations).intersection(mask_rotations)
            if completed_rotations:
                last_rotation = max(completed_rotations)
        except Exception as e:
            print(f"Error determining last processed rotation: {e}")
        
        return last_rotation

    def randomize_object_position(self, obj, boundaries):
        """Randomizes the object's position based on provided boundaries"""
        if obj is None:
            print(f"Warning: Object to randomize not found. Cannot randomize position.")
            return
            
        new_location = []
        for i, bounds in enumerate(boundaries):
            lower, upper = bounds
            # Ensure lower is less than or equal to upper
            if lower > upper:
                lower, upper = upper, lower
                
            # Uniform random value within the bounds
            random_value = np.random.uniform(lower, upper)
            
            new_location.append(random_value)

        obj.location = new_location
        print(f"Randomized position for '{obj.name}' to: {obj.location}")

    def randomize_light_position(self, light_obj):
        """Randomizes the light object's position based on LIGHT_BOUNDARIES"""
        if light_obj is None:
            print(f"Warning: Light object '{LIGHT_OBJECT_NAME}' not found. Cannot randomize position.")
            return

        new_location = []
        for i, bounds in enumerate(LIGHT_BOUNDARIES):
            lower, upper = bounds
            # Ensure lower is less than or equal to upper
            if lower > upper:
                lower, upper = upper, lower
                
            # Uniform random value within the bounds
            random_value = np.random.uniform(lower, upper)
            
            # Randomly pick 1 or -1
            random_sign = np.random.choice([1, -1])
            
            new_location.append(random_value * random_sign)

        light_obj.location = new_location
        print(f"Randomized light position to: {light_obj.location}")

    def execute(self, context):
        print("Starting Database Creation")
        self.setup_output_directories()
        
        # Setup cameras (using global CAMERA_TYPE_FILTER)
        cameras = [{"name": obj.name, "value": obj.name} 
                  for obj in bpy.data.objects if obj.type == CAMERA_TYPE_FILTER]
        if not cameras:
            print("No cameras found in the scene.")
            return {'CANCELLED'}
        
        # Find last processed rotation (now that directories are set up)
        last_rotation = self.find_last_processed_rotation()
        print(f"Resuming from rotation: {last_rotation + 1}")
        
        # Initialize rotation manager (using global TOTAL_ROTATIONS)
        rotation_manager = RotationManager(TOTAL_ROTATIONS)
        
        # Skip to last processed rotation
        for _ in range(last_rotation + 1):
            rotation_manager.get_next_rotation()
        
        # Store state for modal execution
        self.cameras = cameras
        self.rotation_manager = rotation_manager
        self.rotation_index = last_rotation # Initialize rotation index
        self.current_camera_index = 0 # Initialize camera index for the current rotation step
        self.ref_obj = bpy.data.objects.get(REF_OBJECT_NAME)
        self.mask_obj = bpy.data.objects.get(MASK_OBJECT_NAME)
        self.light_obj = bpy.data.objects.get(LIGHT_OBJECT_NAME) # Store the light object

        if not self.ref_obj:
             print(f"Error: Reference object '{REF_OBJECT_NAME}' not found.")
             return {'CANCELLED'}
        if not self.mask_obj:
             print(f"Error: Mask object '{MASK_OBJECT_NAME}' not found.")
             return {'CANCELLED'}
             
        # The modal timer will handle the iteration
        print("Starting modal timer...")
        self.timer_event = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        # Handle cancellation
        if event.type == 'ESC':
            self.cancel(context)
            print("Operation cancelled by user.")
            return {'CANCELLED'}
        
        # Only process on timer events
        if event.type == 'TIMER':
            # Get the current camera for this rotation step
            cam_data = self.cameras[self.current_camera_index]

            # Generate filenames
            base_filename = self.make_filename(self.rotation_index, cam_data["value"])

            image_filepath = str(self.image_path / f"{base_filename}.png")
            json_filepath_ref = str(self.json_path_ref / f"{base_filename}.json")
            json_filepath_mask = str(self.json_path_mask / f"{base_filename}.json")

            # Skip if files already exist
            if os.path.exists(image_filepath) and os.path.exists(json_filepath_ref) and os.path.exists(json_filepath_mask):
                print(f"Files already exist: {base_filename}")
            else:
                # --- Perform the processing step ---
                ref_obj = self.ref_obj
                mask_obj = self.mask_obj
                camera = bpy.data.objects.get(cam_data["value"])
                light_obj = self.light_obj

                # Randomize object position if enabled (only for the first camera of a new rotation)
                if RANDOMIZE_OBJECT_POSITION and self.current_camera_index == 0:
                    self.randomize_object_position(ref_obj, OBJECT_BOUNDARIES)

                # Add object position to JSON data (after potential randomization)
                vertices_data_ref = {
                    'object_position': [ref_obj.location.x, ref_obj.location.y, ref_obj.location.z],
                    'visible_vertices': [],
                    'non_visible_vertices': [],
                    'rotation': {}
                }
                vertices_data_mask = {
                    'object_position': [mask_obj.location.x, mask_obj.location.y, mask_obj.location.z],
                    'visible_vertices': [],
                    'non_visible_vertices': [],
                    'rotation': {}
                }

                # Apply rotation (only for the first camera of a new rotation)
                if self.current_camera_index == 0:
                    rotation = self.rotation_manager.rotations[self.rotation_index] # Get the rotation from the manager
                    self.change_rotation(ref_obj, rotation)
                    self.change_rotation(mask_obj, rotation)

                # Force update of dependency graph to ensure transformations are applied
                bpy.context.view_layer.update()

                # Render
                context.scene.render.filepath = image_filepath
                bpy.ops.render.render(write_still=True)

                # Process vertices and get vertex and rotation data
                vertex_and_rotation_data_ref = self.process_vertices(ref_obj, camera, context.scene)
                vertices_data_ref['visible_vertices'] = vertex_and_rotation_data_ref['visible_vertices']
                vertices_data_ref['non_visible_vertices'] = vertex_and_rotation_data_ref['non_visible_vertices']
                vertices_data_ref['rotation'] = vertex_and_rotation_data_ref['rotation']
                vertices_data_ref['rotation_quaternion'] = vertex_and_rotation_data_ref['rotation_quaternion']

                vertex_and_rotation_data_mask = self.process_vertices(mask_obj, camera, context.scene)
                vertices_data_mask['visible_vertices'] = vertex_and_rotation_data_mask['visible_vertices']
                vertices_data_mask['non_visible_vertices'] = vertex_and_rotation_data_mask['non_visible_vertices']
                vertices_data_mask['rotation'] = vertex_and_rotation_data_mask['rotation']
                vertices_data_mask['rotation_quaternion'] = vertex_and_rotation_data_mask['rotation_quaternion']

                # Randomize light position if enabled
                if RANDOMIZE_LIGHT_POSITION:
                    self.randomize_light_position(light_obj)

                # Add current light position to vertices_data after randomization
                if light_obj:
                    vertices_data_ref['light_position'] = [light_obj.location.x, light_obj.location.y, light_obj.location.z]
                    vertices_data_mask['light_position'] = [light_obj.location.x, light_obj.location.y, light_obj.location.z]
                else:
                    vertices_data_ref['light_position'] = [0.0, 0.0, 0.0]
                    vertices_data_mask['light_position'] = [0.0, 0.0, 0.0]

                # Save JSON data
                with open(json_filepath_ref, 'w') as f:
                    json.dump(vertices_data_ref, f, indent=4)
                with open(json_filepath_mask, 'w') as f:
                    json.dump(vertices_data_mask, f, indent=4)
                # --- End of processing step ---

            # Move to the next camera or next rotation
            self.current_camera_index += 1
            if self.current_camera_index >= len(self.cameras):
                # Finished all cameras for this rotation, move to the next rotation
                self.current_camera_index = 0
                next_rotation = self.rotation_manager.get_next_rotation()
                if next_rotation is None:
                    # Finished all rotations
                    print("Processing completed!")
                    self.cancel(context) # Use cancel for cleanup
                    return {'FINISHED'}
                self.rotation_index = self.rotation_manager.get_rotation_index() # Update rotation index

            # Report progress based on completed rotations
            progress = self.rotation_manager.get_progress()
            print(f"Progress: {progress:.1f}%")

        # Allow other events to be processed
        return {'PASS_THROUGH'}
    
    def cancel(self, context):
        """Clean up timer when operation is cancelled or finished"""
        if self.timer_event:
            context.window_manager.event_timer_remove(self.timer_event)
            self.timer_event = None

def register():
    bpy.utils.register_class(VerticesDatabaseCreator)

def unregister():
    bpy.utils.unregister_class(VerticesDatabaseCreator)


register()
bpy.ops.render.vertices_database()
