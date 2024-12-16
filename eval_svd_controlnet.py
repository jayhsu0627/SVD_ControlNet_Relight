#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py
# Train svd_xtend, tempoeral
# https://github.com/graphdeco-inria/controlnet-diffusers-relighting/blob/main/train_controlnet.py

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
from torch.utils.data import Subset, DataLoader

import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel
from relighting.light_directions import get_light_dir_encoding, BACKWARD_DIR_IDS

from einops import rearrange
# from lpips import LPIPS
import datetime
import diffusers
from diffusers.models.lora import LoRALinearLayer

from diffusers import (
    AsymmetricAutoencoderKL,
    MarigoldDepthPipeline,
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
    StableVideoDiffusionPipeline)

from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from utils.MIL_dataset import MIL
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from models.controlnet_sdv import ControlNetSDVModel

from torch.utils.data import Dataset
from accelerate.utils import DistributedDataParallelKwargs
# from accelerate.utils.deepspeed import get_active_deepspeed_plugin

import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential
from torchvision import transforms
import torchvision.transforms.functional as TF
import kornia
import imageio.v3 as iio

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

# logger = get_logger(__name__, log_level="INFO")

#i should make a utility function file
def validate_and_convert_image(image, target_size=(256, 256)):
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
        # image = image.resize(target_size)
        image = image

    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(512, 512)):
    target_size = images[0].size
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

def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder,i, step_num):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # # Combine frames into a list without converting (since they are already PIL Images)
    # combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(validation_images)
    cols = num_images  # adjust number of columns as needed
    rows = 1

    # Create and save the grid image
    grid_org = create_image_grid(validation_images, rows, cols)
    grid_pre = create_image_grid(flattened_batch_output, rows, cols)
    grid_con = create_image_grid(validation_control_images, rows, cols)

    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    filename_org = f"org_{step_num}_{i}.png"
    filename_pre = f"pre_{step_num}_{i}.png"
    filename_con = f"con_{step_num}_{i}.png"

    output_org_loc = os.path.join(output_folder, filename_org)
    output_pre_loc = os.path.join(output_folder, filename_pre)
    output_con_loc = os.path.join(output_folder, filename_con)

    # print(output_pre_loc)
    if grid_org is not None and step_num==1:
        grid_org.save(output_org_loc)
        grid_con.save(output_con_loc)
    else:
        print("Failed to create image grid")

    if grid_pre is not None:
        # print(output_pre_loc)
        grid_pre.save(output_pre_loc)
    else:
        print("Failed to create image grid")

def resize_and_pad_image(image, target_size=(1024, 512), background_color=(255, 255, 255)):
    
    # Resize the image while preserving aspect ratio
    width, height = image.size
    target_width, target_height = target_size
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    image = image.resize((new_width, new_height), resample=PIL.Image.BICUBIC)
    
    # Convert the resized image to have a transparent background
    image = image.convert("RGBA")
    
    # Create a new image with the target size and white background
    result = Image.new('RGBA', target_size, background_color)
    
    # Paste the resized image into the center of the new image
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    result.paste(image, (x, y), image)
    
    return result.convert('RGB')

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split('_')
        if len(parts) > 1 and parts[0] == 'frame':
            try:
                return int(parts[1].split('.')[0])  # Extracting the number part
            except ValueError:
                return float('inf')  # In case of non-integer part, place this file at the end
        return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)
    print(len(sorted_files))
    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = resize_and_pad_image(img)
            images.append(img)
    return images

# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

def make_train_dataset(args):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = MIL(args.csv_path,
                args.video_folder,
                args.condition_folder,
                args.motion_folder,
                sample_size=args.width,
                sample_n_frames=args.sample_n_frames)
    return dataset

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default='/fs/nexus-scratch/sjxu/Model_out/model_out/checkpoint-8500/controlnet',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--decoder_model_name_or_path",
        type=str,
        default='/fs/nexus-scratch/sjxu/Model_out/decoder/checkpoint-25100/model.safetensors',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=20,
        help=(
            "Run fine-tuning test every X epochs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "path to the dataset csv"
        ),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=(
            "path to the video folder"
        ),
    )
    parser.add_argument(
        "--condition_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        default=None,
        help=(
            "path to the motion folder"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_num",
        type=int,
        default=None,
        help=(
            "num of validation sets"
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=15,
        help=(
            "frames per video"
        ),
    )
    parser.add_argument(
        "--inject_lighting_direction",
        action="store_true",
    )
    parser.add_argument(
        "--concat_depth_maps",
        action="store_true",
    )
    parser.add_argument(
        "--dir_sh",
        type=int, 
        default=-1
    )
    parser.add_argument(
        "--dropout_rgb",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
    )
    parser.add_argument(
        "--target_light",
        type=str,
        default=None,
        help=(
            "insert lighting vector for inference"
        )
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image

def collate_fn(examples, args):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["condition_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()


    input_ids = torch.stack([example["input_ids"] for example in examples])
    target_dir = torch.stack([example["target_dir"] for example in examples])
    motion_values = torch.stack([example["motion_values"] for example in examples])

    result = {
        "pixel_values": pixel_values,
        "condition_pixel_values": conditioning_pixel_values,
        "target_dir": target_dir,
    }

    if args.concat_depth_maps:
        depth_pixel_values = torch.stack([example["depth_pixel_values"] for example in examples])
        depth_pixel_values = depth_pixel_values.to(memory_format=torch.contiguous_format).float()
        result["depth_pixel_values"] = depth_pixel_values

    result["input_ids"] = input_ids
    result["motion_values"] = motion_values
        

    return result

def _load_decoder_pipeline(ckpt_path: str):
    vae = AsymmetricAutoencoderKL.from_config({
        'in_channels': 3, 
        'out_channels': 3, 
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'], 
        'down_block_out_channels': [128, 256, 512, 512], 
        'layers_per_down_block': 2, 
        'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], 
        'up_block_out_channels': [192, 384, 768, 768], 
        'layers_per_up_block': 3, 
        'act_fn': 'silu', 
        'latent_channels': 4, 
        'norm_num_groups': 32, 
        'sample_size': 256, 
        'scaling_factor': 0.18215, 
        'learn_residuals': False, 
        '_use_default_values': ['learn_residuals'], 
        '_class_name': 'AsymmetricAutoencoderKL', 
        '_diffusers_version': '0.19.0.dev0', 
        '_name_or_path': 'cross-attention/asymmetric-autoencoder-kl-x-1-5'
    }).cuda()
    _load_weights(vae, ckpt_path)
    return vae 

def _load_weights(model, ckpt_path):
    from safetensors import safe_open
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device='cuda') as f: 
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    generator = torch.Generator(
        device="cuda").manual_seed(23123134)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # print(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    if args.controlnet_model_name_or_path:
        print("Loading existing controlnet weights")
        controlnet = ControlNetSDVModel.from_pretrained(args.controlnet_model_name_or_path, conditioning_channels=4 if args.concat_depth_maps else 3)
    else:
        print("Initializing controlnet weights from unet")
        controlnet = ControlNetSDVModel.from_unet(unet, conditioning_channels=4 if args.concat_depth_maps else 3)

    if args.decoder_model_name_or_path:
        print("Loading existing vae weights")
        vae = _load_decoder_pipeline(args.decoder_model_name_or_path)
    else:
        print("Initializing vae weights from unet")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant="fp16")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )

    @torch.no_grad()
    def modify_layers(controlnet):
        if args.inject_lighting_direction:
            sample_light_vec = [get_light_dir_encoding(0)] * args.num_frames
            sample_light_vec = torch.from_numpy(np.array(sample_light_vec))
            controlnet.time_embedding.cond_proj = torch.nn.Linear(sample_light_vec.shape[0] * sample_light_vec.shape[2], controlnet.timestep_input_dim, bias=False)
   
    modify_layers(controlnet)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to("cuda", dtype=weight_dtype)
    # text_encoder.to("cuda", dtype=weight_dtype)
    vae.to("cuda", dtype=weight_dtype)
    unet.to("cuda", dtype=weight_dtype)
    # controlnet.to("cuda", dtype=weight_dtype)

    # Inference!
    # total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    text_encoded = None
    image_logs = None

    # The models need unwrapping because for compatibility in distributed training mode.
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        image_encoder=image_encoder,
        vae=vae,
        text_encoder=text_encoder,
        revision=args.revision,
        torch_dtype=weight_dtype,
        insert_light = True if args.inject_lighting_direction else False
    )
    depth_pipeline = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0")
    
    pipeline = pipeline.to("cuda")
    depth_pipeline = depth_pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    # for each loop through the direct underneath
    base_dir = os.path.join(args.validation_image_folder)
    folder_list = os.listdir(base_dir)
    folder_list = sorted(folder_list)

    num_folders = len(folder_list)
    # Prepare condition images (input)
    for i in range(num_folders):
        print(i)
        validation_control_images_cat = []
        
        img_folder = os.path.join(args.validation_image_folder, folder_list[i])
        print(img_folder)
        # validation_image = [load_images_from_folder(img_folder)[0]]
        validation_image = load_images_from_folder(img_folder)[:args.num_frames]

        validation_control_images = load_images_from_folder(img_folder)[:args.num_frames]
        for img in validation_control_images:
            with torch.autocast("cuda"):
                depth_map = depth_pipeline(img).prediction[0]
            img = np.array(img)
            print(img.shape, depth_map.shape)
            img_array = np.concatenate((img, depth_map), axis=2).astype(np.uint8)

            fullres_image = Image.fromarray(img_array)

            # print(fullres_image.shape)
            validation_control_images_cat.append(fullres_image)
            # control_image = F.interpolate(fullres_image, (height, width), mode="bicubic", antialias=True)

        print('num',len(validation_image))
        print('num',len(validation_control_images_cat))

        # run inference
        val_save_dir = os.path.join(
            args.output_dir, folder_list[i])
        print(val_save_dir)

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)

       # Prepare lighting index
        if args.inject_lighting_direction:
            numbers_list = [int(num) for num in args.target_light.split(', ')]
            target_dir = [get_light_dir_encoding(index) for index in numbers_list]
            target_dir = torch.from_numpy(np.array(target_dir))
            print(target_dir.shape)

            # ( t, m, n) to (b, t, m, n)
            target_dir = target_dir.unsqueeze(0)
            target_dir = target_dir.to("cuda")

        w, h = validation_image[0].size
        ratio = w/h
        with torch.autocast(device_type="cuda"):
            video_frames = pipeline(
                validation_image[0], 
                validation_control_images_cat,
                height=args.height,
                width=int(args.height * ratio),
                num_frames=args.num_frames,
                decode_chunk_size=8,
                motion_bucket_id=5,
                fps=args.num_frames,
                noise_aug_strength=0.02,
                train_dir=target_dir,
            ).frames
            
            out_file_path = os.path.join(
                val_save_dir,
                folder_list[i]+ f"_SVD_vae_5500.mp4",
            )
            flattened_batch_output = [img for sublist in video_frames for img in sublist]
            # print(flattened_batch_output[0].shape)

            export_to_gif(flattened_batch_output, out_file_path, 30)    

if __name__ == "__main__":
    main()
ß