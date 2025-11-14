#!/bin/bash

# SmolVLA Reflow Training Script
# Train a 2-Rectified Flow model from an existing teacher model checkpoint

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

# Required: Dataset and teacher model
DATASET_REPO_ID="name/aloha_agix_sim"
DATASET_ROOT="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"

# Teacher model - use the trained checkpoint
TEACHER_MODEL_PATH="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"

# Training configuration
BATCH_SIZE=32
STEPS=20000
LEARNING_RATE=1e-4

# Output
OUTPUT_DIR="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow"

# Hardware
GPU_ID=1

# ============================================================================
# Validation
# ============================================================================

echo "=" | tr '=' '-' | head -c 80; echo
echo "SmolVLA Reflow Training Configuration"
echo "=" | tr '=' '-' | head -c 80; echo

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
echo "✓ Learning rate: $LEARNING_RATE"
echo ""

# ============================================================================
# Training
# ============================================================================

echo "Starting Reflow training..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python src/lerobot/scripts/lerobot_train.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.use_reflow=true \
  --policy.teacher_model_path="$TEACHER_MODEL_PATH" \
  --policy.load_vlm_weights=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=$LEARNING_RATE \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --eval_freq=5000 \
  --save_freq=5000 \
  --output_dir="$OUTPUT_DIR" \
  --wandb.enable=true \
  --wandb.project="aloha_smolvla" \
  --wandb.entity="christianwang-sjtu" \
  --wandb.mode="online" \
  --wandb.notes="Reflow training from checkpoint: $(basename $(dirname $TEACHER_MODEL_PATH))"

echo ""
echo "=" | tr '=' '-' | head -c 80; echo
echo "Training completed! Model saved to: $OUTPUT_DIR"
echo "=" | tr '=' '-' | head -c 80; echo
