#!/bin/bash
# Train SmolVLA model on ALOHA dataset
# This script trains a SmolVLA policy from scratch

# Set environment variables
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp

# Uncomment if using proxy
export http_proxy=http://172.16.0.136:18000 
export https_proxy=http://172.16.0.136:18000

# W&B API Key (uncomment to enable W&B logging)
export WANDB_API_KEY="489fe7b734df1e91930d434d63c36b600b2faed9"

# Generate unique run_id with timestamp
RUN_ID="smolvla_$(date +%Y%m%d_%H%M%S)"

# Training configuration
DATASET_REPO_ID="robotwin/stack_blocks_two"
DATASET_ROOT="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two"
OUTPUT_DIR="/pfs/pfs-ilWc5D/ziqianwang/pretrain_stack_blocks_two"
CUDA_DEVICE="0"
BATCH_SIZE="32"
STEPS="50000"

# Run training
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    --output_dir=${OUTPUT_DIR} \
    --steps=${STEPS} \
    --batch_size=${BATCH_SIZE} \
    --wandb.enable=true \
    --wandb.project=aloha_smolvla \
    --wandb.entity=christianwang-sjtu \
    --wandb.run_id=${RUN_ID} \
    --wandb.mode=online
