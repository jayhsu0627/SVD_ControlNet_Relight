import os
import subprocess

# Path to the directory containing folders with frames
base_dir = "/fs/nexus-scratch/sjxu/WebVid/blender/inference_comb"

# Loop through the folders
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Check if the folder is a valid directory and contains frames (skip non-directories)
    if os.path.isdir(folder_path) and folder.startswith("clip_") and "pred" in folder:
        # Get the clip name, e.g., "clip_01" from "clip_01_pred"
        clip_name = folder.replace("_pred", "")
        output_video_path = os.path.join(base_dir, f"{clip_name}.mp4")
        
        # ffmpeg command to convert images to video
        # Assumes images are named sequentially like frame0001.png, frame0002.png, etc.
        input_pattern = os.path.join(folder_path, "frame_%01d.png")  # Modify pattern as needed
        
        # Example ffmpeg command:
        # ffmpeg -r 30 -i frame%04d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p output.mp4
        ffmpeg_command = [
            "ffmpeg",
            "-r", "15",  # Frame rate (30 fps here, adjust as needed)
            "-i", input_pattern,  # Input file pattern (e.g., frame0001.png, frame0002.png)
            "-c:v", "libx264",  # Video codec
            "-vf", "fps=30",  # Set video frame rate
            "-pix_fmt", "yuv420p",  # Output pixel format for compatibility
            output_video_path
        ]
        
        # Execute the ffmpeg command
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Successfully created video: {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video for {folder}: {e}")
