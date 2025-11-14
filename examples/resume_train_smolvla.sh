#!/bin/bash
# Resume SmolVLA model training from checkpoint
# This script resumes training from a saved checkpoint

# Set environment variables
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp

# Uncomment if using proxy
export http_proxy=http://172.16.0.136:18000
export https_proxy=http://172.16.0.136:18000

# W&B API Key (uncomment to enable W&B logging)
export WANDB_API_KEY="489fe7b734df1e91930d434d63c36b600b2faed9"

# Training configuration
DATASET_REPO_ID="name/aloha_agix_sim"
DATASET_ROOT="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"
OUTPUT_DIR="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin"
CUDA_DEVICE="1"
BATCH_SIZE="32"
STEPS="50000"

# Resume configuration
# IMPORTANT: Set this to the checkpoint directory you want to resume from
# Format: OUTPUT_DIR/checkpoints/step_XXXXXX
# Example: /pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/step_025000
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last"  # Uses the last checkpoint by default

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint directory does not exist: ${CHECKPOINT_PATH}"
    echo "Please set CHECKPOINT_PATH to a valid checkpoint directory."
    echo "Available checkpoints in ${OUTPUT_DIR}/checkpoints:"
    ls -d ${OUTPUT_DIR}/checkpoints/*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Resuming training from checkpoint: ${CHECKPOINT_PATH}"

# Run training with resume flag
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
    --wandb.mode=online \
    --resume=true \
    --checkpoint_path=${CHECKPOINT_PATH}
