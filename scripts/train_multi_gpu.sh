#!/bin/bash

# Multi-GPU Training Launch Script for LeRobot
#
# This script simplifies launching distributed training across multiple GPUs.
# It automatically detects available GPUs and sets up the distributed environment.
#
# Usage:
#   ./scripts/train_multi_gpu.sh [config_path] [additional_args]
#
# Examples:
#   # Train with 4 GPUs
#   ./scripts/train_multi_gpu.sh --config configs/smolvla_aloha.yaml
#
#   # Train with 2 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_multi_gpu.sh --config configs/smolvla_aloha.yaml
#
#   # Train with specific GPUs
#   CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/train_multi_gpu.sh --config configs/smolvla_aloha.yaml

set -e

# Detect number of available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # No CUDA_VISIBLE_DEVICES set, detect all available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Count GPUs from CUDA_VISIBLE_DEVICES
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
fi

echo "==========================================="
echo "LeRobot Multi-GPU Training"
echo "==========================================="
echo "Number of GPUs: $NUM_GPUS"
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi
echo "==========================================="

# Choose the training script
# You can switch between:
# 1. lerobot_train.py (uses Accelerate - recommended)
# 2. lerobot_train_ddp.py (uses native PyTorch DDP)
TRAIN_SCRIPT="src/lerobot/scripts/lerobot_train.py"
# TRAIN_SCRIPT="src/lerobot/scripts/lerobot_train_ddp.py"  # Uncomment to use native DDP

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single GPU mode"
    python $TRAIN_SCRIPT "$@"
else
    echo "Multi-GPU mode with $NUM_GPUS GPUs"

    # Method 1: Using Accelerate (recommended for lerobot_train.py)
    if [[ "$TRAIN_SCRIPT" == *"lerobot_train.py" ]]; then
        echo "Using Accelerate launcher..."
        accelerate launch \
            --num_processes=$NUM_GPUS \
            --mixed_precision=bf16 \
            --multi_gpu \
            $TRAIN_SCRIPT "$@"

    # Method 2: Using torchrun (for lerobot_train_ddp.py)
    else
        echo "Using torchrun launcher..."
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=29500 \
            $TRAIN_SCRIPT "$@"
    fi
fi

echo "==========================================="
echo "Training completed!"
echo "==========================================="
