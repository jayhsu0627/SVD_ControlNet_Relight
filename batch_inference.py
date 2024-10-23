import os
import subprocess

# Directories
base_dir = "/fs/nexus-scratch/sjxu/WebVid/blender"
validation_image_folder_path = os.path.join(base_dir, "img")
validation_control_folder_path = os.path.join(base_dir, "shd")
output_path = os.path.join("/fs/nexus-scratch/sjxu/WebVid/blender/inference")

# Python script that generates images
python_script = "/fs/nexus-scratch/sjxu/svd-temporal-controlnet/run_inference_shd.py"

# Iterate over all folders in validation_image_folder
for image_folder in os.listdir(validation_image_folder_path):
    # Full path to the image folder
    image_folder_path = os.path.join(validation_image_folder_path, image_folder)

    # Skip if it's not a directory
    if not os.path.isdir(image_folder_path):
        continue

    # Get the first image in the folder
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {image_folder_path}")
        continue

    # Sort the images to ensure the "first" one is selected
    image_files.sort()
    validation_image = os.path.join(image_folder_path, image_files[0])

    # Find the corresponding control folder
    control_folder_path = os.path.join(validation_control_folder_path, image_folder)

    # Check if control folder exists
    if not os.path.exists(control_folder_path):
        print(f"Control folder {control_folder_path} does not exist, skipping...")
        continue

    # Prepare arguments for the script
    args = [
        "python", python_script,
        "--validation_image_folder", os.path.join(validation_image_folder_path, image_folder),
        "--validation_control_folder", os.path.join(validation_control_folder_path, image_folder),
        "--validation_image", validation_image,
        "--folder", image_folder,
        "--output_dir",output_path,
    ]

    # Run the script with arguments
    try:
        result = subprocess.run(args, check=True)
        print(f"Successfully processed folder: {image_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing folder {image_folder}: {e}")
