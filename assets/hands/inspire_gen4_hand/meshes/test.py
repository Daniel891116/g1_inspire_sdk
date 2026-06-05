import bpy
import os

# Set paths
input_folder = "/home/soofiyan/workspaces/dextrous_hand/dex-retargeting/assets/robots/hands/inspire_gen4_hand/meshes"  # STL input folder
output_folder = "/home/soofiyan/workspaces/dextrous_hand/dex-retargeting/assets/robots/hands/inspire_gen4_hand/meshes"  # Output folder for GLB files

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each STL file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".stl") or filename.endswith(".STL"):
        # Define file paths
        stl_path = os.path.join(input_folder, filename)
        glb_filename = os.path.splitext(filename)[0] + ".glb"
        glb_path = os.path.join(output_folder, glb_filename)

        # Import the STL file
        bpy.ops.import_mesh.stl(filepath=stl_path)
        
        # Select the imported object and set its origin to geometry
        imported_object = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        # Apply all transforms
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Export the object as GLB
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            use_selection=True,
            export_format='GLB'
        )

        # Delete the object to clean up for the next import
        bpy.data.objects.remove(imported_object, do_unlink=True)

        print(f"Converted {filename} to {glb_filename} with correct origin and transforms")

print("Conversion complete!")