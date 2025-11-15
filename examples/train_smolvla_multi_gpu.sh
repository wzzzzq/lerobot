#!/bin/bash
# Multi-GPU Training for SmolVLA model on ALOHA dataset
# This script trains a SmolVLA policy using multiple GPUs with Accelerate

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

# ============================================
# Multi-GPU Configuration
# ============================================

# Number of GPUs to use
NUM_GPUS=4

# Which GPUs to use (comment out to use all available GPUs)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Mixed precision mode: "no", "fp16", "bf16" (recommended for A100/RTX 3090+)
MIXED_PRECISION="bf16"

# ============================================
# Training Configuration
# ============================================

DATASET_REPO_ID="name/aloha_agix_sim"
DATASET_ROOT="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"
OUTPUT_DIR="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_vlm_multi_gpu"

# Batch size per GPU (effective batch size = BATCH_SIZE × NUM_GPUS)
# Example: 16 per GPU × 4 GPUs = 64 effective batch size
BATCH_SIZE="16"
STEPS="50000"

# Number of data loading workers per GPU
NUM_WORKERS="8"

# ============================================
# Display Configuration
# ============================================

echo "==========================================="
echo "Multi-GPU Training Configuration"
echo "==========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Mixed precision: ${MIXED_PRECISION}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run ID: ${RUN_ID}"
echo "==========================================="

# ============================================
# Run Multi-GPU Training with Accelerate
# ============================================

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=${MIXED_PRECISION} \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.train_expert_only=false \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    --output_dir=${OUTPUT_DIR} \
    --steps=${STEPS} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --wandb.enable=true \
    --wandb.project=aloha_smolvla \
    --wandb.entity=christianwang-sjtu \
    --wandb.run_id=${RUN_ID} \
    --wandb.mode=online

echo "==========================================="
echo "Training completed!"
echo "==========================================="
