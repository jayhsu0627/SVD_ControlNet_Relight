import torch

# # import some helper functions from chrislib (will be installed by the intrinsic repo)
# from chrislib.general import show, view, uninvert
# from chrislib.data_util import load_image
# from chrislib.normal_util import get_omni_normals

# # import model loading and running the pipeline
# from intrinsic.pipeline import run_pipeline, run_gray_pipeline
# from intrinsic.pipeline import load_models
# from pipeline_mod import get_light_coeffs

from PIL import Image 
import numpy as np
import cv2

# import OpenEXR
# import Imath
import numpy
import numexpr as ne
from tqdm.auto import trange
import os
import re
from pathlib import Path

from transformers import pipeline
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import diffusers
import PIL
from PIL import Image
import requests
from skimage.transform import resize

# from boosted_depth.depth_util import create_depth_models, get_depth


# def save_to_rgb(img, directory, file_name, format: str):
#     if format=='nL':
#         temp_result = Image.fromarray(((img+1)/2*255).astype('uint8'))
#     elif format =='compo_shd':
#         temp_result = Image.fromarray((view(compo)*255).astype('uint8')[:,:, 0])
#     else:
#         temp_result = Image.fromarray((view(img)*255).astype('uint8'))
#     temp_result.save(directory + file_name + '_' + format+'.png')

def get_clean_image_list(directory):
    # Get all PNG files
    all_files = [f for f in os.listdir(directory) if f.endswith(('.JPG', '.jpg'))]

    # Filter out files with suffixes like '_mask' or '_skymask'
    clean_files = []
    for file in all_files:
        base_name = file.rsplit('.', 1)[0]  # This handles all extensions
        # Check if this filename contains any underscore
        if 'materials' not in base_name:
            clean_files.append(file)
        else:
            # For cases like '20150415_165607.png' (allow one underscore for timestamp)
            
            parts = base_name.split('_')
            # If it only has one underscore and both parts are numbers
            if len(parts) == 2 and all(part.isdigit() for part in parts):
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

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def depth_to_point_cloud(depth_map, focal_length):
    """
    Convert a depth map into a point cloud in world coordinates.

    Args:
    - depth_map (numpy array): H x W array representing the depth at each pixel
    - focal_length (float): Focal length in pixels

    Returns:
    - point_cloud (numpy array): H x W x 3 array representing the 3D point for each pixel
    """
    print(depth_map.shape)
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Assuming the camera's optical center is at the center of the image
    u = u - W / 2.0
    v = v - H / 2.0

    # Convert depth map to 3D points
    z = depth_map
    x = (u * z) / focal_length
    y = (v * z) / focal_length
    x = -x
    y = -y
    point_cloud = np.stack((x, y, z), axis=-1)  # Shape: H x W x 3
    return point_cloud

def compute_normals_from_point_cloud(point_cloud):
    """
    Compute per-pixel normals from the point cloud using cross products of partial derivatives.

    Args:
    - point_cloud (numpy array): H x W x 3 array representing the 3D point for each pixel

    Returns:
    - normals (numpy array): H x W x 3 array representing the normal vector at each pixel
    """
    # Calculate partial derivatives with respect to x and y
    dz_dx = np.gradient(point_cloud, axis=1)  # Partial derivative w.r.t x
    dz_dy = np.gradient(point_cloud, axis=0)  # Partial derivative w.r.t y

    # Compute the cross product to get the normal vector
    normals = np.cross(dz_dx, dz_dy)

    # Normalize the normals to unit vectors
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= (norm + 1e-8)  # Avoid division by zero

    return normals

def smooth_normals(normals, kernel_size=5):
    """
    Smooth the normal map by averaging the normals of nearby pixels.

    Args:
    - normals (numpy array): H x W x 3 array representing the normal vector at each pixel
    - kernel_size (int): Size of the smoothing kernel

    Returns:
    - smoothed_normals (numpy array): Smoothed normal map
    """
    # Separate the normal components (x, y, z)
    normals_x = normals[:, :, 0]
    normals_y = normals[:, :, 1]
    normals_z = normals[:, :, 2]

    # Apply a smoothing filter (e.g., Gaussian or average) to each component
    smoothed_x = cv2.blur(normals_x, (kernel_size, kernel_size))
    smoothed_y = cv2.blur(normals_y, (kernel_size, kernel_size))
    smoothed_z = cv2.blur(normals_z, (kernel_size, kernel_size))

    # Recombine the smoothed components into a single normal map
    smoothed_normals = np.stack((smoothed_x, smoothed_y, smoothed_z), axis=-1)

    # Normalize the smoothed normals to ensure unit vectors
    norm = np.linalg.norm(smoothed_normals, axis=-1, keepdims=True)
    smoothed_normals /= (norm + 1e-8)  # Avoid division by zero

    return smoothed_normals

def find_median_index(numbers):
    # Sort the list while keeping track of the original indices
    sorted_indices = sorted(range(len(numbers)), key=lambda x: numbers[x])
    
    # Find the median index
    n = len(sorted_indices)
    if n % 2 == 0:
        # For even number of elements, return the index of the lower middle value
        median_index = sorted_indices[n // 2 - 1]
    else:
        # For odd number of elements, return the index of the middle value
        median_index = sorted_indices[n // 2]
    
    return median_index

base_path = '/sdb5/data/train/'

scenes = [f for f in os.listdir(base_path)]
scenes = sorted(scenes)
# # load the models from the given paths
# models = load_models('v2')

# # load pipe
# pipe = pipeline(task="depth-estimation",
#                 # model="depth-anything/Depth-Anything-V2-Large-hf",
#                 model="depth-anything/Depth-Anything-V2-Small-hf",
#                 )

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

pipe_norm = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

# configuration = DepthAnythingConfig()
# configuration.depth_estimation_type = "metric"
# configuration.max_depth = 20

# image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
# model = DepthAnythingForDepthEstimation(configuration)

# print('loading depth model')
# dpt_model = create_depth_models()

# # Load model and preprocessing transform
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

for scene in scenes:
    # scene = '0015'
    # if int(scene)<=17:
    #     # print(scene, ' continued')
    #     continue
    directory_path = base_path + scene

    file_list = get_clean_image_list(directory_path)
    print('======Start process ' + scene + "======")
    # print(file_list)

    depth_condition = Image.open(base_path+ scene +'/' + "all_depth.png")
    width, height = depth_condition.size
    target_width, target_height = 1500, 1000
    print(width, height)
    if width != target_width or height != target_height:
        print(base_path+ scene +'/' + "all_depth.png")

        file_num = 0
        alb_list = []
        luminances = []
        peak_lum = -99
        for file_name in file_list:
            file_path = base_path + scene +'/' + file_name
            image = load_image(file_path)
            # Marigold
            tem_lum = compute_luminance(image)
            luminances.append(tem_lum)
        # Only process the depth for median luminance pic
        median_index = find_median_index(luminances)
            
        image = Image.open(base_path + scene +'/' + file_list[median_index])
        depth = pipe(image)
        normals = pipe_norm(image)
        
        # https://huggingface.co/docs/diffusers/v0.28.2/en/using-diffusers/marigold_usage
        depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
        depth = depth_16bit[0]
        print(depth.size)
        
        depth_array = np.array(depth)
        depth_normalized = depth_array / 65535.0
        depth_8bit = (depth_normalized * 255).astype(np.uint8)

        # Save the normalized depth image as 8-bit
        depth_8bit_image = Image.fromarray(depth_8bit).convert('L')
        foo = depth_8bit_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        foo.save(base_path+ scene +'/' + "all_depth.png")

        print('saved to', base_path+ scene +'/' + "all_depth.png")

        vis = pipe_norm.image_processor.visualize_normals(normals.prediction)

        foo = vis[0].resize((target_width, target_height),Image.Resampling.LANCZOS)
        foo.save(base_path+ scene +'/' + "all_normal.png")

    # for file_name in file_list:

    #     file_path = base_path+ scene +'/data/' + file_name
        
    #     file_pref = file_name.split(sep = ".")[0]
    #     file_suf = file_name.split(sep = ".")[1]

    #     mask_path = base_path+ scene +'/data/'+file_pref+'_skymask.png'

    #     # load an image (np float array in [0-1])
    #     image = load_image(file_path)
    #     mask = load_image(mask_path)
        
    #     bg_h, bg_w = image.shape[:2]
    #     max_dim = max(bg_h, bg_w)
    #     scale = 512 / max_dim

    #     # resize img
    #     small_bg_img = rescale(image, scale)
    #     # small_bg_nrm = get_omni_normals(nrm_model, small_bg_img)

    #     small_bg_shd = cv2.imread(base_path+ scene +'/data/' + file_pref + '_shd.png', cv2.IMREAD_GRAYSCALE)
    #     small_bg_shd = small_bg_shd/255                 # (0,1)
    #     small_bg_shd = rescale(small_bg_shd, scale)
        
    #     normals = cv2.imread(base_path+ scene +'/data/' + 'all_normal.png')
    #     normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)
    #     normals = normals/255                           # (0,1)

    #     small_bg_nrm = rescale(normals, scale)
    #     # small_bg_nrm = (small_bg_nrm + 1) / 2

    #     # we need shd, normal, img, all in resized size
    #     mask = rescale(mask, scale)
        
    #     # make it invert, I don't know why but get_light_coeffs take this as mask
    #     mask[mask>0.5]= int(1)
    #     mask[mask<0.5]= int(0)
    #     mask = -mask+1

    #     # print(mask)
    #     coeffs, temp_shd = get_light_coeffs(small_bg_shd, small_bg_nrm, small_bg_img)
    #     output_img_path = base_path+ scene +'/data/'

    #     save_to_rgb(view(temp_shd)*255, output_img_path, file_pref, 'shd_est')
    #     # # # Output normals for visualization or further processing
    #     # normals_image = ((normals + 1) / 2 * 255).astype(np.uint8)  # Normalize for visualization
    #     # save_to_rgb(normals_image, output_img_path, file_pref, 'normal')

    #     # print(np.array(coeffs[:3]))
    #     # file_num+=1
    #     print("processed ", file_pref)
        
    #     # Save the array as a text file at a specific path
    #     np.savetxt(output_img_path + file_pref+'_light.txt', np.array(coeffs[:3]), fmt='%f')

    # print(luminances)

    # # depth_avg = depth_avg/file_num
    # # save_to_rgb(view(depth_avg)*255, output_img_path, 'all', 'depth')
    # save_to_rgb(depth, output_img_path, 'all', 'depth')
    # # save_to_rgb(view(depth)*255, output_img_path, 'all', 'depth')

    # save_to_rgb(normals_image, output_img_path, 'all', 'normal')

    # save_to_rgb(alb_avg, output_img_path, 'all', 'alb')

    # for file_name in file_list:
    #     file_pref = file_name.split(sep = ".")[0]
    #     file_suf = file_name.split(sep = ".")[1]

    #     frame2 = cv2.imread(base_path+ scene +'/data/' + file_name)[...,::-1]
    #     alb = cv2.imread(base_path+ scene +'/data/' + 'all_alb.png')[...,::-1]

    #     # Compute the shading
    #     shading = compute_shading(frame2, alb)
    #     save_to_rgb(shading, output_img_path , file_pref, 'shd')