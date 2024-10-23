import torch

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, uninvert
from chrislib.data_util import load_image

# import model loading and running the pipeline
from intrinsic.pipeline import run_pipeline, run_gray_pipeline
from intrinsic.pipeline import load_models

from PIL import Image 
import numpy as np
import cv2
# from pipeline import get_light_coeffs

import OpenEXR
import Imath
import numpy
import numexpr as ne
from tqdm.auto import trange
import os
import re
from pathlib import Path

def save_to_rgb(img, directory, file_name, format: str):
    if format=='nL':
        temp_result = Image.fromarray(((img+1)/2*255).astype('uint8'))
    elif format =='compo_shd':
        temp_result = Image.fromarray((view(compo)*255).astype('uint8')[:,:, 0])
    else:
        temp_result = Image.fromarray((view(img)*255).astype('uint8'))
    temp_result.save(directory + file_name + '_' + format+'.png')

def get_clean_image_list(directory):
    # Get all PNG files
    all_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.JPG', '.jpeg', '.tif'))]
    print(all_files)
    # Filter out files with suffixes like '_mask' or '_skymask'
    clean_files = []
    for file in all_files:
        base_name = file.rsplit('.', 1)[0]  # This handles all extensions
        # Check if this filename contains any underscore
        if '_' not in base_name:
            clean_files.append(file)
        else:
            # For cases like '20150415_165607.png' (allow one underscore for timestamp)
            
            parts = base_name.split('_')
            # If it only has one underscore and both parts are numbers
            # if len(parts) == 2 and all(part.isdigit() for part in parts):
            if len(parts) == 2 and all([not('mask' in part or 'sky' in part) for part in parts ]):

                clean_files.append(file)
    
    return sorted(clean_files)

def numpy_interpolate(hr_shd, size, mode='bilinear', align_corners=True, antialias=True):
    # OpenCV uses (width, height) format for resizing, so we need to reverse the size tuple
    target_height, target_width = size
    interpolation = cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST
    
    # Perform interpolation
    resized = cv2.resize(hr_shd, (target_width, target_height), interpolation=interpolation)
    
    return resized

def compute_shading(rendered_image, albedo_image, epsilon=1e-6):
    """
    Computes the shading by dividing the rendered image by the albedo image.

    Parameters:
    - rendered_image: The rendered image (with lighting applied).
    - albedo_image: The albedo image (diffuse color without lighting).
    - epsilon: A small constant to avoid division by zero.

    Returns:
    - shading: The computed shading image.
    """
    # Ensure both images are in float format
    rendered_image = rendered_image.astype(np.float32) / 255.0
    albedo_image = albedo_image.astype(np.float32) / 255.0

    # Avoid division by zero by adding a small epsilon to the denominator
    albedo_image = np.clip(albedo_image, epsilon, 1.0)

    # Compute the shading as rendered / albedo
    shading = rendered_image / albedo_image

    # # Clip the shading values to the range [0, 1] for valid display
    # shading = np.clip(shading, 0.0, 1.0)

    return shading

def compute_luminance(image):
    """
    Compute the luminance of an RGB image using the formula:
    Luminance = 0.299 * R + 0.587 * G + 0.114 * B
    """
    # Convert to float to avoid overflow
    image = image.astype(np.float32)
    luminance = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    return np.mean(luminance)  # Return the mean luminance for weighting

def weighted_average_images(image_list, luminances):
    """
    Compute the weighted average of images using their luminance as weights.

    Args:
    - image_list: List of images (as numpy arrays)
    - luminances: List of corresponding luminance values (used as weights)

    Returns:
    - weighted_avg_image: The final weighted average image
    """
    # Normalize luminances to make sure they sum to 1 (weights)
    weights = np.array(luminances)
    weights = weights / np.sum(weights)
    print(weights)

    # Initialize the weighted sum as zeros
    weighted_sum = np.zeros_like(image_list[0], dtype=np.float32)

    # Loop over each image and apply the corresponding weight
    for img, weight in zip(image_list, weights):
        weighted_sum += img.astype(np.float32) * weight

    # # Convert back to uint8 and return the weighted average image
    # weighted_avg_image = np.clip(weighted_sum, 0, 255).astype(np.uint8)
    # return weighted_avg_image
    return weighted_sum
base_path = '/fs/nexus-scratch/sjxu/bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/'

scenes = [f for f in os.listdir(base_path)]
# load the models from the given paths
models = load_models('v2')

for scene in scenes:
    # scene = '0015'
    if int(scene)<413:
        print(scene, ' continued')
        continue
    directory_path = base_path+ scene +'/data'

    file_list = get_clean_image_list(directory_path)
    print('======Start process ' + scene + "======")
    print(file_list)

    file_num = 0
    alb_list = []
    luminances = []

    for file_name in file_list:
        file_path = base_path+ scene +'/data/' + file_name
        
        file_pref = file_name.split(sep = ".")[0]
        file_suf = file_name.split(sep = ".")[1]

        mask_path = base_path+ scene +'/data/'+file_pref+'_skymask.png'

        # load an image (np float array in [0-1])
        image = load_image(file_path)
        mask = load_image(mask_path)
        orig_sz = image.shape[:2]

        # mask_3d = mask[:, :, np.newaxis]  # Now becomes (M,N,1)
        # mask_3d = np.repeat(mask_3d, 3, axis=2)  # Now becomes (M,N,3)

        # image = image * mask_3d


        # run the model on the image using R_0 resizing
        results = run_pipeline(models, image, resize_conf=0.0)
        # results = run_gray_pipeline(models, image, resize_conf=0.0, maintain_size=True)

        albedo = results['hr_alb']
        albedo = numpy_interpolate(albedo, orig_sz)
        
        # albedo = results['gry_alb']
        diffuse_shading = results['dif_shd']
        diffuse_shading = numpy_interpolate(diffuse_shading, orig_sz)

        # residual = results['residual']

        # if file_num<1: alb_avg = np.zeros_like(albedo)

        # # Calculate the luminance (brightness) using the weighted sum of R, G, B
        # luminance = 0.299 * albedo[:, :, 2] + 0.587 * albedo[:, :, 1] + 0.114 * albedo[:, :, 0]

        # # Calculate the average brightness (mean luminance)
        # brightness = np.mean(luminance)
        luminances.append(compute_luminance(image))

        # print(albedo.shape, alb_avg.shape)
        # print(np.min(alb_avg), np.max(alb_avg))
        # alb_avg += albedo * 1/luminance
        alb_list.append(albedo)

        output_img_path = base_path+ scene +'/data/'
        # + multiple other keys for different intermediate components

        save_to_rgb(albedo, output_img_path, file_pref, 'alb')
        # save_to_rgb(diffuse_shading, output_img_path , file_pref, 'shd')
        # save_to_rgb(residual, output_img_path , file_pref, 'res')

        file_num+=1
        print("processed ", file_pref)
    print(luminances)
    # alb_avg = alb_avg/file_num
    alb_avg = weighted_average_images(alb_list, luminances)

    save_to_rgb(alb_avg, output_img_path, 'all', 'alb')

    for file_name in file_list:
        file_pref = file_name.split(sep = ".")[0]
        file_suf = file_name.split(sep = ".")[1]

        frame2 = cv2.imread(base_path+ scene +'/data/' + file_name)[...,::-1]
        alb = cv2.imread(base_path+ scene +'/data/' + 'all_alb.png')[...,::-1]

        # Compute the shading
        shading = compute_shading(frame2, alb)
        save_to_rgb(shading, output_img_path , file_pref, 'shd')

# import cv2
# import os

# def print_image_shapes(directory_path):
#     # Loop through all files in the directory
#     for filename in os.listdir(directory_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add more extensions as needed
#             image_path = os.path.join(directory_path, filename)
            
#             # Load the image
#             image = cv2.imread(image_path)
            
#             if image is not None:
#                 # Print the shape of the image
#                 print(f"{filename}: {image.shape}")
#             else:
#                 print(f"Could not load {filename}")

# # Example usage
# directory_path = "/fs/nexus-scratch/sjxu/bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/0002/data"  # Replace with your directory path
# print_image_shapes(directory_path)
