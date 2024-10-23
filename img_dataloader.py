import os
import cv2
import torch
from torch.utils.data import Dataset
import glob

class BigTimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Main directory containing subfolders (e.g., /BigTime_v1/).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Collect all subfolders under the root directory
        self.subfolders = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, f))]
        print(self.subfolders)

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        folder_path = self.subfolders[idx]
        data_folder = os.path.join(folder_path, 'data')

        # Glob all PNG files in the data folder
        rgb_files = glob.glob(os.path.join(data_folder, '*.png'))
        
        # Collect valid RGB, skymask, and mask image pairs
        valid_images = []
        fo+r rgb_path in rgb_files:
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            
            # Find corresponding skymask (with same prefix)
            skymask_path = os.path.join(data_folder, base_name + '_skymask.png')
            
            # Proceed only if skymask exists
            if os.path.exists(skymask_path):
                valid_images.append((rgb_path, skymask_path))

        # If no valid images, raise an error
        if not valid_images:
            raise ValueError(f"No valid image pairs found in {folder_path}")
        print(valid_images)

        # Process all valid pairs
        processed_images = []
        for rgb_image_path, skymask_path in valid_images:
            # Load RGB image
            rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
            
            # Load skymask image
            skymask = cv2.imread(skymask_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure skymask is binary (values 0 or 1)
            skymask = skymask / 255.0

            # Apply skymask to the RGB image (set sky pixels to 0)
            masked_image = rgb_image * skymask[:, :, None]  # Mask each color channel

            # If any transform is provided, apply it to the masked image
            if self.transform:
                masked_image = self.transform(masked_image)

            # Collect the processed images
            processed_images.append((masked_image, rgb_image_path))

        # Return the list of processed images and paths
        return processed_images

# Example of how to use the dataset
if __name__ == "__main__":
    root_dir = "/fs/nexus-scratch/sjxu/bigtime/phoenix/S6/zl548/AMOS/BigTime_v1/"
    dataset = BigTimeDataset(root_dir=root_dir)
    print(len(dataset))
    print(dataset.__getitem__)


    # # Example: iterate through the dataset and process the images
    # for i, (masked_image, image_path) in enumerate(dataset):
    #     # Do additional processing or save the result
    #     output_path = f'/processed_output/masked_image_{i}.png'
    #     cv2.imwrite(output_path, masked_image)
    #     print(f"Processed and saved: {output_path}")
