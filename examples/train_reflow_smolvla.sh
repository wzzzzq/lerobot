#!/bin/bash

# SmolVLA Reflow Training Script
# This script demonstrates how to train a SmolVLA model using Reflow

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Dataset configuration
DATASET_REPO_ID="your_dataset_repo_id"  # Replace with your dataset

# Model configuration
TEACHER_MODEL_PATH="lerobot/smolvla_base"  # Can be HF repo or local path
VLM_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Training configuration
BATCH_SIZE=32
STEPS=50000
EVAL_FREQ=5000
SAVE_FREQ=5000

# Reflow configuration
REFLOW_NUM_STEPS=10  # Number of ODE steps for generating X_1

# Output configuration
OUTPUT_DIR="outputs/smolvla_2rf"

# Hardware configuration
GPU_ID=0

# ============================================================================
# Training
# ============================================================================

echo "============================================"
echo "SmolVLA Reflow Training"
echo "============================================"
echo "Dataset: $DATASET_REPO_ID"
echo "Teacher Model: $TEACHER_MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path="$TEACHER_MODEL_PATH" \
  --policy.reflow_num_inference_steps=$REFLOW_NUM_STEPS \
  --policy.vlm_model_name="$VLM_MODEL" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.expert_width_multiplier=0.75 \
  --policy.num_steps=10 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --output_dir="$OUTPUT_DIR" \
  --wandb.enable=true \
  --wandb.project="smolvla_reflow" \
  --wandb.name="reflow_2rf"

echo "============================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "============================================"

# ============================================================================
# Optional: Evaluation with different inference steps
# ============================================================================

echo ""
echo "Running evaluation with different inference steps..."

for NUM_STEPS in 2 5 10; do
  echo "Evaluating with num_steps=$NUM_STEPS..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/eval.py \
    --policy.path="$OUTPUT_DIR" \
    --policy.num_steps=$NUM_STEPS \
    --env.name="your_env_name" \
    --output_file="$OUTPUT_DIR/eval_${NUM_STEPS}steps.json"
done

echo "============================================"
echo "All evaluations completed!"
echo "============================================"
