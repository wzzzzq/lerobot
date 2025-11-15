#!/bin/bash
# Multi-GPU Training Template for LeRobot
#
# This is a template script for multi-GPU training using Accelerate.
# Customize the configuration section below for your specific task.

set -e  # Exit on error

# ============================================
# GPU Configuration
# ============================================

# Number of GPUs to use
NUM_GPUS=4

# Select specific GPUs (optional)
# Uncomment and modify to use specific GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Mixed precision: "no", "fp16", "bf16"
# Recommendation:
#   - "bf16" for Ampere GPUs (A100, RTX 3090, RTX 4090, etc.)
#   - "fp16" for older GPUs (V100, RTX 2080 Ti, etc.)
#   - "no" to disable mixed precision
MIXED_PRECISION="bf16"

# ============================================
# Training Configuration
# ============================================

# Policy configuration
POLICY_PATH="lerobot/smolvla_base"  # Pretrained model path or policy type
TRAIN_EXPERT_ONLY=false              # Set to true to only train the action expert

# Dataset configuration
DATASET_REPO_ID="your-username/your-dataset"
DATASET_ROOT=""  # Leave empty to download from HuggingFace Hub

# Training parameters
BATCH_SIZE=16     # Batch size PER GPU (effective batch size = BATCH_SIZE × NUM_GPUS)
STEPS=50000       # Total training steps
NUM_WORKERS=8     # Data loading workers PER GPU

# Output configuration
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"

# ============================================
# WandB Configuration (Optional)
# ============================================

ENABLE_WANDB=true
WANDB_PROJECT="robot-learning"
WANDB_ENTITY="your-username"
WANDB_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"

# Set W&B API key (or use: wandb login)
# export WANDB_API_KEY="your-wandb-api-key"

# ============================================
# Environment Variables (Optional)
# ============================================

# Disable NCCL P2P and InfiniBand if you encounter issues
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

# Set NCCL debug level for troubleshooting
# export NCCL_DEBUG=INFO

# Proxy settings (if needed)
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port

# ============================================
# Display Configuration
# ============================================

echo "==========================================="
echo "Multi-GPU Training Configuration"
echo "==========================================="
echo "Policy: ${POLICY_PATH}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Mixed precision: ${MIXED_PRECISION}"
echo "Total steps: ${STEPS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "WandB enabled: ${ENABLE_WANDB}"
if [ "${ENABLE_WANDB}" = true ]; then
    echo "WandB project: ${WANDB_PROJECT}"
    echo "WandB run ID: ${WANDB_RUN_ID}"
fi
echo "==========================================="

# ============================================
# Check if accelerate is installed
# ============================================

if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate is not installed. Please run: pip install accelerate"
    exit 1
fi

# ============================================
# Run Multi-GPU Training
# ============================================

# Build the accelerate command
ACCELERATE_CMD="accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=${MIXED_PRECISION} \
    --multi_gpu"

# Build the training command
TRAIN_CMD="src/lerobot/scripts/lerobot_train.py \
    --policy.path=${POLICY_PATH} \
    --policy.push_to_hub=false \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --output_dir=${OUTPUT_DIR} \
    --steps=${STEPS} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --wandb.enable=${ENABLE_WANDB} \
    --wandb.project=${WANDB_PROJECT} \
    --wandb.entity=${WANDB_ENTITY} \
    --wandb.run_id=${WANDB_RUN_ID}"

# Add dataset root if specified
if [ -n "${DATASET_ROOT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --dataset.root=${DATASET_ROOT}"
fi

# Execute training
echo ""
echo "Starting training..."
echo ""

${ACCELERATE_CMD} ${TRAIN_CMD}

# ============================================
# Post-training
# ============================================

echo ""
echo "==========================================="
echo "Training completed successfully!"
echo "==========================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""

# Optional: Run evaluation after training
# echo "Running evaluation..."
# python src/lerobot/scripts/lerobot_eval.py \
#     --policy.path=${OUTPUT_DIR}/checkpoints/last \
#     --eval.n_episodes=10

echo "Done!"
