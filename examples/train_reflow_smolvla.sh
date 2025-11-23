#!/bin/bash

# SmolVLA Reflow Training Script (Clean Architecture)
# Train a 2-Rectified Flow model from an existing teacher model checkpoint
#
# Clean Reflow Architecture:
# - Reflow is a training method, NOT a model architecture
# - SmolVLA model code is COMPLETELY CLEAN - zero reflow-specific code
# - Reflow logic is ONLY in lerobot_train_reflow.py training script
# - Teacher model is instantiated in training script, not in model class
# - Student model starts from teacher's weights, learns straight-line flows
#
# Key Benefits:
# - Complete separation of concerns: model vs training method
# - SmolVLA doesn't know how it's being trained (Flow Matching principle)
# - No duplicate preprocessing: observations preprocessed once for both models
# - Teacher and student use identical model architecture
# - Checkpoint is fully compatible with standard inference

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
REFLOW_LR="1e-7"  # 5x smaller than normal training

# Training configuration
BATCH_SIZE=32
STEPS=100000

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

echo "Starting Reflow training with clean architecture..."
echo ""
echo "Clean Architecture Design:"
echo "  - SmolVLA model: 100% reflow-agnostic (no reflow code in model)"
echo "  - Training script: Handles all reflow logic"
echo "  - Teacher: Instantiated in training script, frozen for X_0 generation"
echo "  - Student: Initialized from teacher checkpoint, fine-tuned with straight-line targets"
echo "  - Preprocessing: Done once, shared by both teacher and student"
echo "  - No duplicate prepare_images/prepare_state calls"
echo ""

# Use lerobot_train_reflow.py which implements clean reflow architecture
# The script automatically:
#   1. Loads teacher from teacher_model_path (frozen)
#   2. Initializes student from same checkpoint (trainable)
#   3. Automatically freezes VLM and vision encoder (only trains expert)
#   4. Generates X_0 via teacher.model.sample_actions(noise=X_1)
#   5. Trains student with X_0 as target (straight-line flows)
CUDA_VISIBLE_DEVICES=$GPU_ID python src/lerobot/scripts/lerobot_train_reflow.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --teacher_model_path="$TEACHER_MODEL_PATH" \
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
  --wandb.notes="Reflow training (clean architecture) from checkpoint: $(basename $(dirname $TEACHER_MODEL_PATH))"

echo ""
echo "================================================================================"
echo "Training completed! Model saved to: $OUTPUT_DIR"
echo "================================================================================"
echo ""
echo "Clean Architecture Benefits:"
echo "  - Model code: 100% reflow-agnostic (no training method leakage)"
echo "  - Checkpoint: Standard SmolVLAPolicy, no reflow traces"
echo "  - Inference: SmolVLAPolicy.from_pretrained('$OUTPUT_DIR/checkpoints/...')"
echo "  - Performance: ~2x faster training (no duplicate preprocessing)"
echo "  - Maintainability: Clear separation between model and training method"
echo ""
echo "Technical Details:"
echo "  - Time parameterization: t=0 is data, t=1 is noise (same as standard FM)"
echo "  - Integration direction: t=1 → t=0 (noise to data, same as standard FM)"
echo "  - Only difference from FM: actions come from teacher instead of dataset"
