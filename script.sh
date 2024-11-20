#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=24:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4

# NCCL configuration
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ens3f0  # or the relevant network interface name
export HOST_GPU_NUM=8
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
srun accelerate launch train_svd_con.py \
 --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="/fs/nexus-scratch/sjxu/Model_out/model_out" \
 --csv_path="/fs/nexus-scratch/sjxu/WebVid/blender.csv" \
 --video_folder="/fs/nexus-scratch/sjxu/WebVid/blender/img" \
 --condition_folder="/fs/nexus-scratch/sjxu/WebVid/blender/shd" \
 --motion_folder="/fs/nexus-scratch/sjxu/WebVid/blender/motion" \
 --validation_image_folder="/fs/nexus-scratch/sjxu/svd-temporal-controlnet/validation_demo/img_blender" \
 --validation_control_folder="/fs/nexus-scratch/sjxu/svd-temporal-controlnet/validation_demo/shd_blender" \
 --width=512 \
 --height=512 \
 --learning_rate=1e-4 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=30 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=8 \
 --checkpointing_steps=2000 \
 --validation_steps=200 \
 --gradient_checkpointing \
 --checkpoints_total_limit=1 \
 --report_to="wandb" \
 --dropout_rgb=0.1 \
 --sample_n_frames=5 \
 --num_frames=5





