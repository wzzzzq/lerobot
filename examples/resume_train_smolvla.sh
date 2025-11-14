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
STEPS="100000"

# Resume configuration
# IMPORTANT: When using --resume true, you need to specify:
# 1. --output_dir: Directory containing the checkpoints folder
# 2. --config_path: Path to train_config.json from the checkpoint you want to resume from

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -d ${OUTPUT_DIR}/checkpoints/*/ | grep -v "last" | sort -V | tail -1)
LATEST_CHECKPOINT=$(basename ${LATEST_CHECKPOINT})

# Construct config path
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/${LATEST_CHECKPOINT}/pretrained_model/train_config.json"

# Verify paths exist
if [ ! -d "${OUTPUT_DIR}/checkpoints" ]; then
    echo "Error: Checkpoints directory does not exist: ${OUTPUT_DIR}/checkpoints"
    echo "Please set OUTPUT_DIR to a directory containing a checkpoints folder."
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: train_config.json not found at: ${CONFIG_PATH}"
    echo "Available checkpoints:"
    ls -d ${OUTPUT_DIR}/checkpoints/*/ 2>/dev/null
    exit 1
fi

echo "Available checkpoints:"
ls -d ${OUTPUT_DIR}/checkpoints/*/ 2>/dev/null
echo ""
echo "Resuming training from checkpoint: ${LATEST_CHECKPOINT}"
echo "Using config: ${CONFIG_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Run training with resume flag
# NOTE: You must specify both --resume true and --config_path
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python src/lerobot/scripts/lerobot_train.py \
    --config_path=${CONFIG_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --steps=${STEPS} \
    --resume=true
