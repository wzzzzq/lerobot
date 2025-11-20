#!/bin/bash

# SmolVLA Reflow Training Script (New Architecture)
# Train a 2-Rectified Flow model from an existing teacher model checkpoint
#
# New Reflow Architecture:
# - Reflow is a training method, not a separate model
# - training_mode="reflow" enables reflow loss calculation
# - Teacher model is loaded at training time and set to policy.model.teacher
# - Same SmolVLAPolicy class is used for both standard and reflow training
#
# Key Benefits:
# - Conceptually clearer: Training method vs Model type
# - Single Policy class: No SmolVLAReflowPolicy needed
# - Easier to maintain: All reflow logic in modeling_smolvla.py
# - Simpler configuration: Just set training_mode="reflow"

set -e  # Exit on error

# ============================================================================
# Configuration - Modify these variables for your setup
# ============================================================================

# Environment variables
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp
export http_proxy=http://172.16.0.136:18000
export https_proxy=http://172.16.0.136:18000
export WANDB_API_KEY="489fe7b734df1e91930d434d63c36b600b2faed9"

# Cache and temporary directories
export WANDB_DIR="/pfs/pfs-ilWc5D/ziqianwang/wandb"
export WANDB_CACHE_DIR="/pfs/pfs-ilWc5D/ziqianwang/wandb_cache"
export WANDB_DATA_DIR="/pfs/pfs-ilWc5D/ziqianwang/wandb_data"
export HF_HOME="/pfs/pfs-ilWc5D/ziqianwang/huggingface"
export TRANSFORMERS_CACHE="/pfs/pfs-ilWc5D/ziqianwang/huggingface/transformers"
export HF_DATASETS_CACHE="/pfs/pfs-ilWc5D/ziqianwang/huggingface/datasets"
export TORCH_HOME="/pfs/pfs-ilWc5D/ziqianwang/torch"
export TRITON_CACHE_DIR="/pfs/pfs-ilWc5D/ziqianwang/triton_cache"

# Required: Dataset and teacher model
DATASET_REPO_ID="name/aloha_agix_sim"
DATASET_ROOT="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"

# Teacher model - use the trained checkpoint
TEACHER_MODEL_PATH="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"

# ============================================================================
# Reflow Training Hyperparameters
# ============================================================================

# Learning rate for reflow training (should be smaller than normal training)
# - Normal training: 1e-4 (from scratch)
# - Reflow training: 1e-5 to 5e-5 (fine-tuning from teacher)
# - Reflow is "straightening" trajectories, not learning from scratch
# - Lower LR prevents disrupting already-learned features
REFLOW_LR="2e-5"  # 5x smaller than normal training

# Training configuration
BATCH_SIZE=32
STEPS=20000

# Output
OUTPUT_DIR="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow_new"

# Hardware
GPU_ID=2

# ============================================================================
# Preparation - Create necessary directories
# ============================================================================

echo "Creating cache directories..."
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/tmp
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/wandb
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/wandb_cache
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/wandb_data
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/huggingface
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/torch
mkdir -p /pfs/pfs-ilWc5D/ziqianwang/triton_cache

# ============================================================================
# Validation
# ============================================================================

echo "================================================================================"
echo "SmolVLA Reflow Training Configuration (New Architecture)"
echo "================================================================================"

# Check if teacher model exists
if [ ! -d "$TEACHER_MODEL_PATH" ]; then
    echo "Error: Teacher model not found at: $TEACHER_MODEL_PATH"
    echo "Please set TEACHER_MODEL_PATH to a valid checkpoint directory."
    echo ""
    echo "Expected structure:"
    echo "  $TEACHER_MODEL_PATH/"
    echo "    ├── config.json"
    echo "    ├── model.safetensors"
    echo "    └── ..."
    exit 1
fi

echo "✓ Teacher model found: $TEACHER_MODEL_PATH"
echo "✓ Dataset: $DATASET_REPO_ID"
echo "✓ Output directory: $OUTPUT_DIR"
echo "✓ GPU: $GPU_ID"
echo "✓ Batch size: $BATCH_SIZE"
echo "✓ Steps: $STEPS"
echo "✓ Learning rate: $REFLOW_LR (smaller than standard 1e-4)"
echo ""

# ============================================================================
# Training
# ============================================================================

echo "Starting Reflow training with new architecture..."
echo ""
echo "Key differences from old implementation:"
echo "  - training_mode='reflow' enables reflow loss"
echo "  - teacher_model_path is passed as a training argument"
echo "  - Uses single SmolVLAPolicy class (no SmolVLAReflowPolicy)"
echo "  - Teacher model loaded in training script"
echo ""

# Note: We need to load teacher model and set it to policy.model.teacher
# This is typically done in the training script modification
# For now, we use the same lerobot_train.py with additional argument
CUDA_VISIBLE_DEVICES=$GPU_ID python src/lerobot/scripts/lerobot_train_reflow.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.training_mode=reflow \
  --policy.teacher_model_path="$TEACHER_MODEL_PATH" \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.optimizer_lr="$REFLOW_LR" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --save_freq=5000 \
  --output_dir="$OUTPUT_DIR" \
  --wandb.enable=true \
  --wandb.project="aloha_smolvla_reflow" \
  --wandb.entity="christianwang-sjtu" \
  --wandb.mode="online" \
  --wandb.notes="Reflow training (new architecture) from checkpoint: $(basename $(dirname $TEACHER_MODEL_PATH))"

echo ""
echo "================================================================================"
echo "Training completed! Model saved to: $OUTPUT_DIR"
echo "================================================================================"
echo ""
echo "Notes:"
echo "  - Reflow is a training method, not a separate model"
echo "  - Checkpoint is fully compatible with standard SmolVLAPolicy"
echo "  - Load for inference: SmolVLAPolicy.from_pretrained('$OUTPUT_DIR/checkpoints/...')"
echo "  - No special reflow config needed for inference"
