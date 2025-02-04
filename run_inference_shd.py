import os
import torch
import datetime
import numpy as np
from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from models.controlnet_sdv import ControlNetSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re 
import GPUtil
from threading import Thread
import time
import os
import argparse

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        

def save_gifs_side_by_side(batch_output, validation_images, validation_control_images, output_folder):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    for idx, image_list in enumerate([validation_images, validation_control_images, flattened_batch_output]):
        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        gifs = [Image.open(gif) for gif in gif_paths]
    
        # Find the minimum frame count among all GIFs
        min_frames = min(gif.n_frames for gif in gifs)
    
        frames = []
        for frame_idx in range(min_frames):
            combined_frame = None
            for gif in gifs:
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
    
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=gifs[0].info['duration'])


    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{timestamp}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # # Clean up temporary GIFs
    # for gif_path in gif_paths:
    #     os.remove(gif_path)

    return combined_gif_path

# Define functions
def validate_and_convert_image(image, target_size=(512, 512)):
# def validate_and_convert_image(image, target_size=(1024, 576)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
        # image = image
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]

    # Combine frames into a list
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(256, 256))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        # Try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images

# Main script
if __name__ == "__main__":
    args = {
        "pretrained_model_name_or_path": "stabilityai/stable-video-diffusion-img2vid",
        "validation_image_folder": "./validation_demo/img_blender_ran",
        "validation_control_folder": "./validation_demo/shd_blender_ran",
        "validation_image": "./validation_demo/img_blender_ran/full_rgb_l_frame0300.png",
        "folder": "clip",
        "output_dir": "./output",
        "height": 512,
        "width": 512,
        # cant be bothered to add the args in myself, just use notepad
    }

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some images.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid", help="Path to the pretrained model")
    parser.add_argument("--validation_image_folder", type=str, default="./validation_demo/img_blender_ran", help="Path to the validation image folder")
    parser.add_argument("--validation_control_folder", type=str, default="./validation_demo/shd_blender_ran", help="Path to the validation control folder")
    parser.add_argument("--validation_image", type=str, default="./validation_demo/img_blender_ran/full_rgb_l_frame0300.png", help="Path to the specific validation image")
    parser.add_argument("--folder", type=str, default="clip", help="Name of the folder for output or processing")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--height", type=int, default=512, help="Height of the output image")
    parser.add_argument("--width", type=int, default=512, help="Width of the output image")
    
    # Parse the arguments
    args_parser = parser.parse_args()


    # Instantiate monitor with a 10-second delay between updates
    monitor = Monitor(10)

    # Load validation images and control images
    validation_images = load_images_from_folder_to_pil(args_parser.validation_image_folder)
    #validation_images = convert_list_bgra_to_rgba(validation_images)
    validation_control_images = load_images_from_folder_to_pil(args_parser.validation_control_folder)
    validation_image = Image.open(args_parser.validation_image).convert('RGB')


    # Load and set up the pipeline
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("CiaraRowles/temporal-controlnet-depth-svd-v1",subfolder="controlnet")
    controlnet = controlnet = ControlNetSDVModel.from_pretrained("/fs/nexus-scratch/sjxu/Model_out/SVD_shd/model_out",subfolder="controlnet")

    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"],subfolder="unet")
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args["pretrained_model_name_or_path"],controlnet=controlnet,unet=unet)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()

    # Create output directory if it doesn't exist
    val_save_dir = os.path.join(args_parser.output_dir, args_parser.folder)
    print(args_parser.validation_image_folder)
    print("Saving to ", val_save_dir)
    os.makedirs(val_save_dir, exist_ok=True)

    # Inference and saving loop
    #print(validation_control_images.shape)
    video_frames = pipeline(validation_image, 
                            validation_control_images[:12], 
                            decode_chunk_size=8,
                            num_frames=12,
                            motion_bucket_id=5,
                            controlnet_cond_scale=1.0,
                            width=512,
                            height=512).frames

    # save_gifs_side_by_side(video_frames,validation_images, validation_control_images,val_save_dir)
    flattened_batch_output = [img for sublist in video_frames for img in sublist]
    pil_frames = [validate_and_convert_image(img) for img in flattened_batch_output]
    for idx, image in enumerate(pil_frames):
        path = os.path.join(val_save_dir, f"frame_{idx}.png")
        # img = Image.fromarray(image)
        img = image
        # Save as PNG
        img.save(path)

    monitor.stop()
