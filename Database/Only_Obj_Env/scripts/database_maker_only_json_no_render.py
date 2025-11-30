import bpy
import math
import mathutils
import json
import os
import numpy as np
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
import time

#HARDCODED VALUES
ONLY_JSON = True # DONT CHANGE THIS VALUE
np.random.seed(2)
TOTAL_ROTATIONS = 100 # Total number of random rotations to generate

BASE_OUTPUT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONLYJSON_SUBDIR = "1M" # Subdirectory for JSON files when ONLY_JSON is True

DEBUG_MATERIALS = False
RANDOMIZE_LIGHT_POSITION = False
LIGHT_BOUNDARIES = [[11, 15], # X
                    [11, 15], # Y
                    [11, 15]] # Z


RANDOMIZE_OBJECT_POSITION = True
OBJECT_BOUNDARIES = [[-0.1, 0.1], # X
                     [-0.1, 0.1], # Y
                     [-0.1, 0.1]] # Z

EPSILON = 1e-5
OBJECT_TO_ROTATE_NAME = "REF_TANGO"
CAMERA_TYPE_FILTER = 'CAMERA'
LIGHT_OBJECT_NAME = "Sun" 


# Utility functions
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
    
    def setup_output_directories(self):
        # Use global base path and subdir names
        self.base_path = Path(BASE_OUTPUT_PATH)
        self.only_json_path = self.base_path / ONLYJSON_SUBDIR

        # Create ONLYJSON_SUBDIR if ONLY_JSON is enabled
        if ONLY_JSON:
            self.only_json_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {self.only_json_path}")
    
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
        return f"rotation_{rotation_index:04d}_camera_{camera_name}"
    
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
        
        if ONLY_JSON:
            # If ONLY_JSON, check for JSON files in the ONLYJSON_SUBDIR
            pattern = "rotation_*_camera_*.json"
            existing_files = list(self.only_json_path.glob(pattern))
        
        for file in existing_files:
            try:
                rotation = int(file.stem.split('_')[1])
                last_rotation = max(last_rotation, rotation)
            except:
                continue
        
        return last_rotation

    def randomize_object_position(self, obj, boundaries):
        """Randomizes the object's position based on provided boundaries"""
        if obj is None:
            print(f"Warning: Object to rotate '{OBJECT_TO_ROTATE_NAME}' not found. Cannot randomize position.")
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
        print(f"Randomized object position to: {obj.location}")

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
        
        # Find last processed rotation
        last_rotation = self.find_last_processed_rotation()
        print(f"Resuming from rotation: {last_rotation + 1}")
        
        # Setup cameras (using global CAMERA_TYPE_FILTER)
        cameras = [{"name": obj.name, "value": obj.name} 
                  for obj in bpy.data.objects if obj.type == CAMERA_TYPE_FILTER]
        if not cameras:
            print("No cameras found in the scene.")
            return {'CANCELLED'}
        
        # Initialize rotation manager (using global TOTAL_ROTATIONS)
        rotation_manager = RotationManager(TOTAL_ROTATIONS)
        
        # Skip to last processed rotation
        for _ in range(last_rotation + 1):
            rotation_manager.get_next_rotation()
        
        # Process rotations
        while True:
            rotation = rotation_manager.get_next_rotation()
            if rotation is None:
                break
                
            rotation_index = rotation_manager.get_rotation_index()
            
            for cam in cameras:
                # Generate filenames
                base_filename = self.make_filename(rotation_index, cam["value"])
                
                if ONLY_JSON:
                    # If ONLY_JSON, only construct the JSON filepath in the new subdir
                    image_filepath = None # No image rendered
                    json_filepath = str(self.only_json_path / f"{base_filename}.json")
                
                # Skip if files already exist (check only JSON if ONLY_JSON is true)
                if (ONLY_JSON and os.path.exists(json_filepath)) or \
                   (not ONLY_JSON and os.path.exists(image_filepath) and os.path.exists(json_filepath)):
                    print(f"Files already exist: {base_filename}")
                    continue
                
                # Process the rotation (using global OBJECT_TO_ROTATE_NAME)
                obj = bpy.data.objects.get(OBJECT_TO_ROTATE_NAME)
                camera = bpy.data.objects.get(cam["value"])
                light_obj = bpy.data.objects.get(LIGHT_OBJECT_NAME)
                
                # Randomize object position if enabled
                if RANDOMIZE_OBJECT_POSITION:
                    self.randomize_object_position(obj, OBJECT_BOUNDARIES)

                # Add object position to JSON data (after potential randomization)
                # Note: Vertex coordinates in JSON are relative to world origin,
                # but this records the object's origin for reference.
                vertices_data = {
                    'object_position': [obj.location.x, obj.location.y, obj.location.z],
                    'visible_vertices': [], # Placeholder, filled by process_vertices
                    'non_visible_vertices': [], # Placeholder, filled by process_vertices
                    'rotation': {} # Placeholder, filled by process_vertices
                }

                # Apply rotation
                self.change_rotation(obj, rotation)
                
                # Force update of dependency graph to ensure transformations are applied
                bpy.context.view_layer.update()
                
                # Render if ONLY_JSON is False
                if not ONLY_JSON:
                    # Render and save the still image when not ONLY_JSON
                    context.scene.render.filepath = image_filepath
                    bpy.ops.render.render(write_still=True)
                
                # Process vertices and get vertex and rotation data
                # Note: This data is based on the object's position and rotation at this point.
                vertex_and_rotation_data = self.process_vertices(obj, camera, context.scene)
                
                # Merge vertex and rotation data into the main vertices_data dictionary
                vertices_data['visible_vertices'] = vertex_and_rotation_data['visible_vertices']
                vertices_data['non_visible_vertices'] = vertex_and_rotation_data['non_visible_vertices']
                vertices_data['rotation'] = vertex_and_rotation_data['rotation']
                vertices_data['rotation_quaternion'] = vertex_and_rotation_data['rotation_quaternion']
                
                # Randomize light position if enabled
                if RANDOMIZE_LIGHT_POSITION:
                    self.randomize_light_position(light_obj)
                    
                # Add current light position to vertices_data after randomization
                if light_obj:
                    vertices_data['light_position'] = [light_obj.location.x, light_obj.location.y, light_obj.location.z]
                else:
                    vertices_data['light_position'] = [0.0, 0.0, 0.0] # Default or handle case where light is not found
                
                # Save JSON data
                with open(json_filepath, 'w') as f:
                    json.dump(vertices_data, f, indent=4)
                
                # Report progress
                progress = rotation_manager.get_progress()
                print(f"Progress: {progress:.1f}%")
                
                # Add a small delay to prevent crashes during rapid operations
                if not ONLY_JSON:
                    time.sleep(0.1)
        
        print("Processing completed!")
        return {'FINISHED'}
    
    def modal(self, context, event):
        if event.type == 'ESC':
            print("Cancelling operation...")
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}

def register():
    bpy.utils.register_class(VerticesDatabaseCreator)

def unregister():
    bpy.utils.unregister_class(VerticesDatabaseCreator)


register()
bpy.ops.render.vertices_database()