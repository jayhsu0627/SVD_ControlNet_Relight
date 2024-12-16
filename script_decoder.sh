#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=24:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4

# NCCL configuration
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ens3f0  # or the relevant network interface name
export HOST_GPU_NUM=4
export NCCL_IB_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
srun accelerate launch train_svd_decoder.py \
 --output_dir="/fs/nexus-scratch/sjxu/Model_out/decoder" \
 --width=512 \
 --height=512 \
 --learning_rate=1e-4 \
 --per_gpu_batch_size=4 \
 --num_train_epochs=30 \
 --mixed_precision="bf16" \
 --gradient_accumulation_steps=4 \
 --checkpointing_steps=100 \
 --validation_steps=200 \
 --gradient_checkpointing \
 --checkpoints_total_limit=5 \
 --report_to="wandb" \
 --num_workers=4 \
 --mse_weight=0.4 \
 --vae_model_name_or_path="/fs/nexus-scratch/sjxu/Model_out/decoder/checkpoint-25100/" \
 --resume_from_checkpoint="/fs/nexus-scratch/sjxu/Model_out/decoder/checkpoint-25100"
#  --controlnet_model_name_or_path="/fs/nexus-scratch/sjxu/Model_out/model_out/checkpoint-4500/controlnet" \
#  --resume_from_checkpoint="/fs/nexus-scratch/sjxu/Model_out/model_out/checkpoint-4500"