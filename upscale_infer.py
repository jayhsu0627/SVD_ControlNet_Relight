import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

import GPUtil
from threading import Thread
import time
import os
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from accelerate.inference import prepare_pippy

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

monitor = Monitor(20)

# load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
model_id = "stabilityai/sd-x2-latent-upscaler"

# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

distributed_state = PartialState()

# pipeline = pipeline.to("cuda")
pipeline.to(distributed_state.device)

# pipeline.enable_attention_slicing()
# pipeline.to(distributed_state.device)
# pipeline.enable_attention_slicing()
pipeline.set_use_memory_efficient_attention_xformers(True)
# pipeline.enable_vae_tiling()
pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
pipeline.enable_attention_slicing()
pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)

# pipeline.enable_sequential_cpu_offload()

# # let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)

low_res_img = Image.open("/fs/nexus-scratch/sjxu/svd-temporal-controlnet/test_clip.png").convert("RGB")
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))

prompt = "a classroom"

with distributed_state.split_between_processes(prompt) as prompt:
    result = pipeline(prompt=prompt, image=low_res_img).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
    result.save(f"/fs/nexus-scratch/sjxu/svd-temporal-controlnet/result_{distributed_state.process_index}.png")

monitor.stop()