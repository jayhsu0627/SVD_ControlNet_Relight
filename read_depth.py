from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the depth image
depth_image = Image.open("/fs/gamma-projects/svd_relight/MIT/train/willow_basement30/all_depth.png")
depth_array = np.array(depth_image)

# Normalize the depth for visualization
depth_normalized = depth_array / 65535.0
depth_8bit = (depth_normalized * 255).astype(np.uint8)

# Save the normalized depth image as 8-bit
depth_8bit_image = Image.fromarray(depth_8bit)
depth_8bit_image.save("/fs/nexus-scratch/sjxu/svd-temporal-controlnet/depth_8bit.png")
