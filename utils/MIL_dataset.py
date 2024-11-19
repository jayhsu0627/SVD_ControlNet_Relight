import os, io, csv, math, random, json
import glob
import numpy as np
from einops import rearrange
import re
from collections import defaultdict

import torch
from decord import VideoReader
import cv2

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import sys
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image, ImageOps
import imageio.v3 as iio
import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential

def save_array_as_image(array, filename):
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        print('yeah')
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    print('shape:', array.shape)
    array = array.transpose(1, 2, 0)  

    # Convert the array to an image using PIL
    img = Image.fromarray(array)
    
    # Save the image to the specified filename
    img.save(filename)

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255

    
    #         image = Image.open(image_path).convert("RGB")
    #         probe_path = os.path.dirname(args.images_dir) + "/probes_exr/" + name.replace(".png", "_probe.png")  

    #         conditioning_image_path = args.images_dir + "/" + self.json[i]["conditioning_image"].replace(".jpg", ".png")
    #         conditioning_image = Image.open(conditioning_image_path).convert("RGB")
    #         target_dir = get_light_dir_encoding(int(image_path.split("_dir_")[-1].replace(".png", "")))

    #         result = {
    #             "text": "",
    #             "image": image,
    #             "target_dir": target_dir,
    #             "conditioning_image": conditioning_image,
    #             "pixel_values": TF.to_tensor(image) * 2 - 1,
    #             "conditioning_pixel_values": TF.to_tensor(conditioning_image),
    #             "input_ids": INPUT_IDS,
    #             **(self.xtras)
    #         }
    #         assert "_exr" in image_path #!!! update this
    #         depth_image = Image.open(image_path.replace("_exr/", "_depths/").split("_dir")[0] + "_dir_10_pred.png").convert("L") #!!! update this
    #         result["depth_image"] = depth_image
    #         result["depth_pixel_values"] = TF.to_tensor(depth_image)

    #     except UnidentifiedImageError:
    #         print(f"WARNING: invalid file for scene {image_path}, will return another random batch item")
    #         return super().__getitem__(random.choice(list(range(len(self)))))
    #     return result 

class MIL(Dataset):
    def __init__(
            self, csv_path, video_folder,condition_folder,motion_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):

        self.json = [
            json.loads(line) for line in open(f"relighting/training.json", "r").read().splitlines()
        ]
        print(self.json[0])

        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        # print(self.dataset)
        # print(self.json)
        self.dataset = self.json
        # self.length = len(self.dataset)
        self.length = len(self.json)

        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        self.condition_folder = condition_folder
        self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        sample_width = sample_size
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.groups = {
            "A": range(0, 5),
            "B": range(5, 10),
            "C": range(10, 15),
            "D": range(15, 20),
            "E": range(20, 25)
        }

        # Crop operation
        self.transforms_0 = ImageSequential(
            K.RandomCrop((sample_width, sample_width)),
            same_on_batch=True  # This enables getting the transformation matrices
        )

        # Blue operation
        self.transforms_1 = ImageSequential(
            # K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=(-5., 5.), padding_mode=(0), p=0.7),
            K.RandomPerspective(distortion_scale=0.2, p=0.7),
            K.RandomFisheye(
                center_x = torch.tensor([-.1, .1]),
                center_y = torch.tensor([-.1, .1]),
                gamma = torch.tensor([.95, 1.05]),
                p=0.7,
            ),
            same_on_batch=True  # This enables getting the transformation matrices
        )

        # # Red operation
        # self.transforms_2 = ImageSequential(
        #     K.ColorJitter(brightness=0.0, contrast=0., saturation=0.1, hue=0.1, p=0.5),
        #     K.ColorJiggle(brightness=0.0, contrast=0., saturation=0.1, hue=0.1, p=0.5),
        #     same_on_batch=True  # This enables getting the transformation matrices
        # )

    def __len__(self):
        return self.length

    def sort_frames(self, frame_name):
        # Extract the numeric part from the filename
        # dir_0_mip2.jpg
        frame_name = frame_name.split('.')[0]
        parts = frame_name.split('_')
        # print('parts', parts)
        if len(parts) > 2:
            return int(parts[1])
        else:
            return 9999
            # return int(parts[1][5:])

    def group_filenames_by_set(self, filenames, video_id):
        # Prepare a dictionary to hold filtered filenames by group
        grouped_files = defaultdict(list)
        
        # Pattern to extract frame number from filename (e.g., 14n_office8_dir_0.jpg -> 0)
        pattern = re.compile(f"dir_(\d+)")
        # Iterate over filenames and match them to groups
        for filename in filenames:
            match = pattern.search(filename)
            if match:
                frame_num = int(match.group(1))  # Extract the frame number
                
                # Check which group this frame number belongs to and add it to the group
                for group_name, frame_range in self.groups.items():
                    if frame_num in frame_range:
                        grouped_files[group_name].append(filename)
                        break  # Stop once we find the correct group
        
        return grouped_files

    def get_batch(self, idx):        
        # for blender random

        while True:
            video_dict = self.dataset[idx]
            videoid = video_dict['video_id']
            print(videoid)

            video_id = self.json[idx]["video_id"].replace(".jpg", ".png")
            target_set = self.json[idx]["target_image"].replace(".jpg", ".png")
            cond_set = self.json[idx]["conditioning_image"].replace(".jpg", ".png")

            print(video_id, target_set, cond_set)
            self.video_folder = '/fs/gamma-projects/svd_relight/MIT/train'
            image_path = self.video_folder + "/" + video_id

            # self.video_folder = '/fs/gamma-projects/svd_relight/MIT/train'
            # image_path = self.video_folder + "/" + video_id

            print(image_path)
            filenames = sorted([f for f in os.listdir(image_path) if f.endswith(".jpg")], key=self.sort_frames)[:self.sample_n_frames]
            # video_id = "14n_office8"
            print(filenames)

            grouped_files = self.group_filenames_by_set(filenames, video_id)
            print(grouped_files)
            # Display the grouped files
            for group_name, files in grouped_files.items():
                if group_name == target_set:
                    print(f"target_set {group_name}: {files}")
                    image_files = files
                if group_name == cond_set:
                    print(f"cond_set {group_name}: {files}")
                    cond_files = files

            # preprocessed_dir = os.path.join(self.video_folder, videoid)
            # condition_folder = os.path.join(self.condition_folder, videoid)
            # print(self.motion_values_folder, videoid)
            # motion_values_file = os.path.join(self.motion_values_folder, videoid, videoid + "_average_motion.txt")

            # if not os.path.exists(condition_folder) or not os.path.exists(motion_values_file):
            #     idx = random.randint(0, len(self.dataset) - 1)
            #     continue
    
            # # Sort and limit the number of image and depth files to 14
            # image_files = sorted(os.listdir(preprocessed_dir), key=self.sort_frames)[:self.sample_n_frames]
            # depth_files = sorted(os.listdir(condition_folder), key=self.sort_frames)[:self.sample_n_frames]

            # # Check if there are enough frames for both image and depth
            # if len(image_files) < self.sample_n_frames or len(depth_files) < self.sample_n_frames:
            #     idx = random.randint(0, len(self.dataset) - 1)
            #     continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(image_path, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
    
            # Load depth frames
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(image_path, cond))) for cond in cond_files])
            cond_pixel_values = numpy_to_pt(numpy_depth_images)
    
            # # Load motion values
            # with open(motion_values_file, 'r') as file:
            #     motion_values = float(file.read().strip())

            motion_values = [5]*len(cond_files)
            batch_size = pixel_values.shape[0]

            combined = self.transforms_0(torch.cat([pixel_values, cond_pixel_values], dim=0))
            combined = self.transforms_1(combined)

            pixel_values, cond_pixel_values = combined[:batch_size], combined[batch_size:]
            return pixel_values, cond_pixel_values, motion_values

    def __getitem__(self, idx):
        
        pixel_values, depth_pixel_values,motion_values = self.get_batch(idx)
        pixel_values = self.pixel_transforms(pixel_values)

        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values,motion_values=motion_values)
        return sample




if __name__ == "__main__":
    from utils.util import save_videos_grid

    dataset = MIL(
        csv_path="/fs/nexus-scratch/sjxu/WebVid/blender_random.csv",
        video_folder="/fs/nexus-scratch/sjxu/WebVid/blender_random/img",
        condition_folder = "/fs/nexus-scratch/sjxu/WebVid/blender_random/shd",
        motion_folder = "/fs/nexus-scratch/sjxu/WebVid/blender_random/motion",
        sample_size=512,
        sample_stride=4, sample_n_frames=25
        )

    idx = np.random.randint(len(dataset))
    train_image, train_cond, _ = dataset.get_batch(idx)
    print('length:', len(dataset))
    print(train_image.shape, train_image.shape)

    save_array_as_image(train_image[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_0.png')
    save_array_as_image(train_cond[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_0.png')

    save_array_as_image(train_image[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_1.png')
    save_array_as_image(train_cond[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_1.png')

    save_array_as_image(train_image[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_2.png')
    save_array_as_image(train_cond[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_2.png')

    # import pdb
    # pdb.set_trace()
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    # for idx, batch in enumerate(dataloader):
    #     print(batch["pixel_values"].shape, len(batch["text"]))
    #     print(batch["depth_pixel_values"].shape, len(batch["text"]))
    #     # for i in range(batch["pixel_values"].shape[0]):
    #     #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)