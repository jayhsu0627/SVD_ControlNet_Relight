import torch

# import some helper functions from chrislib (will be installed by the intrinsic repo)
from chrislib.general import show, view, uninvert
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals

# import model loading and running the pipeline
from intrinsic.pipeline import run_pipeline, run_gray_pipeline
from intrinsic.pipeline import load_models
from pipeline import get_light_coeffs

from PIL import Image 
import numpy as np
import cv2

import OpenEXR
import Imath
import numpy
import numexpr as ne
from tqdm.auto import trange
import os
import re
from pathlib import Path

from transformers import pipeline
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
from PIL import Image
import requests
from skimage.transform import resize

from boosted_depth.depth_util import create_depth_models, get_depth


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
    all_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.JPG', '.jpg', '.jpeg', '.tif'))]

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
base_path = '/fs/nexus-scratch/sjxu/bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/'

scenes = [f for f in os.listdir(base_path)]
# load the models from the given paths
models = load_models('v2')

# load pipe
configuration = DepthAnythingConfig()
pipe = pipeline(task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                )

for scene in scenes:
    # scene = '0015'
    if int(scene)!=36:
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
        
        bg_h, bg_w = image.shape[:2]
        max_dim = max(bg_h, bg_w)
        scale = 512 / max_dim

        # resize img

        small_bg_img = rescale(image, scale)
        # small_bg_nrm = get_omni_normals(nrm_model, small_bg_img)

        small_bg_shd = cv2.imread(base_path+ scene +'/data/' + file_pref + '_shd.png', cv2.IMREAD_GRAYSCALE)
        small_bg_shd = rescale(small_bg_shd, scale)


        # load image
        image = Image.open(file_path)

        # calculate nrm from depth
        # Depth-anything-V2
        depth = pipe(image)["depth"]
        depth = np.array(depth)

        # Set camera parameters
        focal_length = 500.0  # Focal length in pixels

        # Step 1: Convert depth map to point cloud
        point_cloud = depth_to_point_cloud(depth, focal_length)

        # Step 2: Compute normals from point cloud
        normals = compute_normals_from_point_cloud(point_cloud)
        small_bg_nrm = rescale(normals, scale)
        small_bg_nrm = (small_bg_nrm + 1) / 2


        #     # Step 2: Visualize the point cloud
        # plot_point_cloud(point_cloud)

        # we need shd, normal, img, all in resized size
        mask = rescale(mask, scale)
        
        # make it invert, I don't know why but get_light_coeffs take this as mask
        mask = -mask+1 
        coeffs, _ = get_light_coeffs(small_bg_shd, small_bg_nrm, small_bg_img, mask=mask)
        output_img_path = base_path+ scene +'/data/'

        save_to_rgb(depth, output_img_path, file_pref, 'depth')
        # # Output normals for visualization or further processing
        normals_image = ((normals + 1) / 2 * 255).astype(np.uint8)  # Normalize for visualization
        save_to_rgb(normals_image, output_img_path, file_pref, 'normal')

        print(coeffs[:3])
        file_num+=1
        print("processed ", file_pref)
    
    # print(luminances)

    # alb_avg = weighted_average_images(alb_list, luminances)

    # save_to_rgb(alb_avg, output_img_path, 'all', 'alb')

    # for file_name in file_list:
    #     file_pref = file_name.split(sep = ".")[0]
    #     file_suf = file_name.split(sep = ".")[1]

    #     frame2 = cv2.imread(base_path+ scene +'/data/' + file_name)[...,::-1]
    #     alb = cv2.imread(base_path+ scene +'/data/' + 'all_alb.png')[...,::-1]

    #     # Compute the shading
    #     shading = compute_shading(frame2, alb)
    #     save_to_rgb(shading, output_img_path , file_pref, 'shd')




# import numpy as np
# import cv2
# import plotly.graph_objects as go



# # Example usage
# # def main():
# # Load a sample depth map (replace with your own depth map)
# # depth_map = cv2.imread('/Users/shengjiexu/Downloads/depth/video_1/output_0001.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
# depth_map = cv2.imread('/Users/shengjiexu/Downloads/depth/video_1/output_0012.png',cv2.IMREAD_GRAYSCALE)

# # Set camera parameters
# focal_length = 500.0  # Focal length in pixels

# # Step 1: Convert depth map to point cloud
# point_cloud = depth_to_point_cloud(depth_map, focal_length)
# # print(point_cloud)

# # Step 2: Compute normals from point cloud
# normals = compute_normals_from_point_cloud(point_cloud)

# # Output normals for visualization or further processing
# normals_image = ((normals + 1) / 2 * 255).astype(np.uint8)  # Normalize for visualization
#     # Step 2: Visualize the point cloud
# plot_point_cloud(point_cloud)

# # # cv2.imwrite('normals.png', normals_image)
# # # Display the occlusion mask
# # plt.imshow(normals_image)
# # plt.axis('off')
# # plt.show()

# # if __name__ == '__main__':
# #     main()
