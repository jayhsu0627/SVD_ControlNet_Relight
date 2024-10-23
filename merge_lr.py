import os
from PIL import Image

# Directory containing left and right frame folders
inference_dir = "/fs/nexus-scratch/sjxu/WebVid/blender/inference"
output_dir = "/fs/nexus-scratch/sjxu/WebVid/blender/inference_comb"
# Loop through the inference directory
for folder in os.listdir(inference_dir):
    # Find left and right folders by checking the folder name
    if "left" in folder:
        clip_name = folder.split("_left_")[0]  # Extract clip name (e.g., "clip_01")
        right_folder = folder.replace("left", "right")
        
        # Construct full paths for left and right folders
        left_folder_path = os.path.join(inference_dir, folder)
        right_folder_path = os.path.join(inference_dir, right_folder)

        # Check if corresponding right folder exists
        if not os.path.exists(right_folder_path):
            print(f"Right folder {right_folder_path} does not exist, skipping...")
            continue

        # Create output folder for combined frames
        output_folder = os.path.join(output_dir, f"{clip_name}_pred")
        os.makedirs(output_folder, exist_ok=True)

        # Loop through left folder images
        for left_frame_name in sorted(os.listdir(left_folder_path)):
            if left_frame_name.endswith(('.png', '.jpg', '.jpeg')):
                # Full path to left and right frames
                left_frame_path = os.path.join(left_folder_path, left_frame_name)
                right_frame_path = os.path.join(right_folder_path, left_frame_name)

                # Ensure the right frame exists
                if not os.path.exists(right_frame_path):
                    print(f"Matching right frame {right_frame_path} not found for {left_frame_name}, skipping...")
                    continue

                # Open both left and right images
                left_img = Image.open(left_frame_path)
                right_img = Image.open(right_frame_path)

                # Resize images to (512, 512) if needed
                left_img = left_img.resize((512, 512))
                right_img = right_img.resize((512, 512))

                # Create a blank canvas for the full frame (1024x512)
                full_frame = Image.new("RGB", (1024, 512))

                # Paste left and right images into the canvas
                full_frame.paste(left_img, (0, 0))     # Paste left image on the left
                full_frame.paste(right_img, (512, 0))  # Paste right image on the right

                # Save the combined image in the output folder
                output_frame_path = os.path.join(output_folder, left_frame_name)
                full_frame.save(output_frame_path)

                print(f"Saved combined frame as {output_frame_path}")
