#!/bin/bash

# Complete SmolVLA Reflow Training Pipeline
# This script runs the full pipeline: base training → Reflow → evaluation

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Dataset configuration
DATASET_REPO_ID="your_dataset_repo_id"  # Replace with your dataset

# Model configuration
VLM_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Output configuration
OUTPUT_BASE="outputs"
OUTPUT_1RF="$OUTPUT_BASE/smolvla_1rf"
OUTPUT_2RF="$OUTPUT_BASE/smolvla_2rf"
OUTPUT_3RF="$OUTPUT_BASE/smolvla_3rf"  # Optional

# Training configuration
BASE_STEPS=100000
REFLOW_STEPS=50000
BATCH_SIZE=32

# Hardware configuration
GPU_ID=0

# Control flags
TRAIN_BASE=true      # Stage 1: Train base model (1-RF)
TRAIN_2RF=true       # Stage 2: Train 2-Rectified Flow
TRAIN_3RF=false      # Stage 3: Train 3-Rectified Flow (optional)
RUN_EVAL=true        # Run evaluation after training

# ============================================================================
# Stage 1: Train Base Model (1-RF)
# ============================================================================

if [ "$TRAIN_BASE" = true ]; then
  echo ""
  echo "========================================"
  echo "Stage 1: Training Base Model (1-RF)"
  echo "========================================"
  echo "Output: $OUTPUT_1RF"
  echo "Steps: $BASE_STEPS"
  echo ""

  CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/train.py \
    --policy.type=smolvla \
    --policy.vlm_model_name="$VLM_MODEL" \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.expert_width_multiplier=0.75 \
    --policy.num_steps=10 \
    --policy.optimizer_lr=1e-4 \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --batch_size=$BATCH_SIZE \
    --steps=$BASE_STEPS \
    --eval_freq=10000 \
    --save_freq=10000 \
    --output_dir="$OUTPUT_1RF" \
    --wandb.enable=true \
    --wandb.project="smolvla_reflow_pipeline" \
    --wandb.name="1rf_base"

  echo ""
  echo "Stage 1 completed! Model saved to $OUTPUT_1RF"
  echo ""
fi

# ============================================================================
# Stage 2: Reflow Training (2-RF)
# ============================================================================

if [ "$TRAIN_2RF" = true ]; then
  echo ""
  echo "========================================"
  echo "Stage 2: Reflow Training (2-RF)"
  echo "========================================"
  echo "Teacher: $OUTPUT_1RF"
  echo "Output: $OUTPUT_2RF"
  echo "Steps: $REFLOW_STEPS"
  echo ""

  CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/train.py \
    --policy.type=smolvla \
    --policy.use_reflow=true \
    --policy.teacher_model_path="$OUTPUT_1RF" \
    --policy.reflow_num_inference_steps=10 \
    --policy.vlm_model_name="$VLM_MODEL" \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.expert_width_multiplier=0.75 \
    --policy.num_steps=10 \
    --policy.optimizer_lr=1e-4 \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --batch_size=$BATCH_SIZE \
    --steps=$REFLOW_STEPS \
    --eval_freq=5000 \
    --save_freq=5000 \
    --output_dir="$OUTPUT_2RF" \
    --wandb.enable=true \
    --wandb.project="smolvla_reflow_pipeline" \
    --wandb.name="2rf_reflow"

  echo ""
  echo "Stage 2 completed! Model saved to $OUTPUT_2RF"
  echo ""
fi

# ============================================================================
# Stage 3: Second Reflow Training (3-RF) - Optional
# ============================================================================

if [ "$TRAIN_3RF" = true ]; then
  echo ""
  echo "========================================"
  echo "Stage 3: Second Reflow Training (3-RF)"
  echo "========================================"
  echo "Teacher: $OUTPUT_2RF"
  echo "Output: $OUTPUT_3RF"
  echo "Steps: $REFLOW_STEPS"
  echo ""

  CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/train.py \
    --policy.type=smolvla \
    --policy.use_reflow=true \
    --policy.teacher_model_path="$OUTPUT_2RF" \
    --policy.reflow_num_inference_steps=10 \
    --policy.vlm_model_name="$VLM_MODEL" \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.expert_width_multiplier=0.75 \
    --policy.num_steps=10 \
    --policy.optimizer_lr=5e-5 \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --batch_size=$BATCH_SIZE \
    --steps=$REFLOW_STEPS \
    --eval_freq=5000 \
    --save_freq=5000 \
    --output_dir="$OUTPUT_3RF" \
    --wandb.enable=true \
    --wandb.project="smolvla_reflow_pipeline" \
    --wandb.name="3rf_reflow2"

  echo ""
  echo "Stage 3 completed! Model saved to $OUTPUT_3RF"
  echo ""
fi

# ============================================================================
# Evaluation: Compare Models with Different Inference Steps
# ============================================================================

if [ "$RUN_EVAL" = true ]; then
  echo ""
  echo "========================================"
  echo "Evaluation: Comparing Models"
  echo "========================================"
  echo ""

  # Create evaluation directory
  EVAL_DIR="$OUTPUT_BASE/evaluation_results"
  mkdir -p "$EVAL_DIR"

  # Evaluation environment (replace with your environment)
  ENV_NAME="your_env_name"

  # Evaluate 1-RF with different steps
  if [ -d "$OUTPUT_1RF" ]; then
    echo "Evaluating 1-RF..."
    for NUM_STEPS in 2 5 10 20; do
      echo "  - num_steps=$NUM_STEPS"
      CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/eval.py \
        --policy.path="$OUTPUT_1RF" \
        --policy.num_steps=$NUM_STEPS \
        --env.name="$ENV_NAME" \
        --output_file="$EVAL_DIR/1rf_${NUM_STEPS}steps.json" || true
    done
  fi

  # Evaluate 2-RF with different steps
  if [ -d "$OUTPUT_2RF" ]; then
    echo "Evaluating 2-RF..."
    for NUM_STEPS in 2 5 10 20; do
      echo "  - num_steps=$NUM_STEPS"
      CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/eval.py \
        --policy.path="$OUTPUT_2RF" \
        --policy.num_steps=$NUM_STEPS \
        --env.name="$ENV_NAME" \
        --output_file="$EVAL_DIR/2rf_${NUM_STEPS}steps.json" || true
    done
  fi

  # Evaluate 3-RF with different steps (if trained)
  if [ -d "$OUTPUT_3RF" ]; then
    echo "Evaluating 3-RF..."
    for NUM_STEPS in 2 5 10 20; do
      echo "  - num_steps=$NUM_STEPS"
      CUDA_VISIBLE_DEVICES=$GPU_ID python lerobot/scripts/eval.py \
        --policy.path="$OUTPUT_3RF" \
        --policy.num_steps=$NUM_STEPS \
        --env.name="$ENV_NAME" \
        --output_file="$EVAL_DIR/3rf_${NUM_STEPS}steps.json" || true
    done
  fi

  echo ""
  echo "Evaluation completed! Results saved to $EVAL_DIR"
  echo ""

  # Generate comparison table (if Python is available)
  echo "Generating comparison table..."
  python3 - <<EOF || true
import json
import os
from pathlib import Path

eval_dir = Path("$EVAL_DIR")
results = {}

# Load all evaluation results
for json_file in eval_dir.glob("*.json"):
    model_name = json_file.stem  # e.g., "1rf_10steps"
    try:
        with open(json_file) as f:
            data = json.load(f)
            results[model_name] = data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")

# Print comparison table
print("\n" + "="*60)
print("Model Performance Comparison")
print("="*60)
print(f"{'Model':<20} {'Steps':<8} {'Success Rate':<15} {'Avg Reward':<15}")
print("-"*60)

for model_name in sorted(results.keys()):
    model_type = model_name.split('_')[0]  # 1rf, 2rf, 3rf
    num_steps = model_name.split('_')[1].replace('steps', '')
    data = results[model_name]

    success_rate = data.get('success_rate', 'N/A')
    avg_reward = data.get('avg_reward', 'N/A')

    if isinstance(success_rate, (int, float)):
        success_rate = f"{success_rate:.2%}"
    if isinstance(avg_reward, (int, float)):
        avg_reward = f"{avg_reward:.3f}"

    print(f"{model_type.upper():<20} {num_steps:<8} {success_rate:<15} {avg_reward:<15}")

print("="*60)
EOF

fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Trained models:"
[ -d "$OUTPUT_1RF" ] && echo "  ✓ 1-RF: $OUTPUT_1RF"
[ -d "$OUTPUT_2RF" ] && echo "  ✓ 2-RF: $OUTPUT_2RF"
[ -d "$OUTPUT_3RF" ] && echo "  ✓ 3-RF: $OUTPUT_3RF"
echo ""
echo "Next steps:"
echo "  1. Check evaluation results in $EVAL_DIR"
echo "  2. Compare models at different inference steps"
echo "  3. Deploy the best model for your application"
echo "  4. (Optional) Perform distillation for single-step inference"
echo ""
echo "========================================"
