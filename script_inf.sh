#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=01:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# NCCL configuration
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ens3f0  # or the relevant network interface name
export HOST_GPU_NUM=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=8

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export NUMEXPR_MAX_THREADS=16
echo "MASTER_ADDR="$MASTER_ADDR

## run
srun python eval_svd_controlnet.py \
 --validation_image_folder="/fs/nexus-scratch/sjxu/controlnet-diffusers-relighting/exemplars/" \
 --output_dir="/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output" \
 --concat_depth_maps \
 --width=512 \
 --height=256 \
 --mixed_precision="bf16" \
 --inject_lighting_direction \
 --target_light='23, 0, 1, 18, 19' \
 --num_frames=5 \
 --controlnet_model_name_or_path='/fs/nexus-scratch/sjxu/Model_out/model_add_ligh_3/checkpoint-4000/controlnet' \
 --multi_frame_inference \
 --enable_xformers_memory_efficient_attention \
 --allow_tf32 \
 --use_8bit_adam
#  --controlnet_model_name_or_path='/fs/nexus-scratch/sjxu/Model_out/model_add_ligh_2_change_HD/checkpoint-5000/controlnet' \
#  --decoder_model_name_or_path='/fs/nexus-scratch/sjxu/Model_out/decoder/checkpoint-50200/decoder/diffusion_pytorch_model.safetensors' \
#  --decoder_model_name_or_path='/fs/nexus-scratch/sjxu/controlnet-diffusers-relighting/weights/decoder_1536x1024.safetensors'
#  --decoder_model_name_or_path='/fs/nexus-scratch/sjxu/Model_out/decoder/checkpoint-50200/decoder/diffusion_pytorch_model.safetensors' \
#  --target_light='23, 0, 1, 18, 19' \
#  --target_light='18, 18, 18, 18, 18' \
