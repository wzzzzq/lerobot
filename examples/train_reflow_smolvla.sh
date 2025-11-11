#!/bin/bash

# SmolVLA Reflow Training Script
# Train a 2-Rectified Flow model from an existing teacher model

set -e  # Exit on error

# ============================================================================
# Configuration - Modify these variables for your setup
# ============================================================================

# Required: Dataset and teacher model
DATASET_REPO_ID="your_dataset_repo_id"              # Your training dataset
TEACHER_MODEL_PATH="lerobot/smolvla_base"           # HF repo or local path

# Training configuration
BATCH_SIZE=32
STEPS=50000
LEARNING_RATE=1e-4

# Output
OUTPUT_DIR="outputs/smolvla_2rf"

# Hardware
GPU_ID=0

# ============================================================================
# Training
# ============================================================================

echo "Starting Reflow training..."
echo "Teacher: $TEACHER_MODEL_PATH"
echo "Dataset: $DATASET_REPO_ID"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path="$TEACHER_MODEL_PATH" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.optimizer_lr=$LEARNING_RATE \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --eval_freq=5000 \
  --save_freq=5000 \
  --output_dir="$OUTPUT_DIR" \
  --wandb.enable=true \
  --wandb.project="smolvla_reflow" \
  --wandb.name="2rf"

echo "Training completed! Model saved to: $OUTPUT_DIR"
