#!/bin/bash
# Resume training from checkpoint (generic template)
# This script can be adapted for any policy type

# Usage:
#   1. Set the CHECKPOINT_PATH to the checkpoint directory you want to resume from
#   2. Run: bash examples/resume_train.sh
#
# Note: All other training parameters (dataset, policy type, batch size, etc.)
# will be loaded from the checkpoint's config.yaml automatically.

# Set environment variables (adjust as needed)
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# IMPORTANT: Set this to the checkpoint directory you want to resume from
# Format: /path/to/output_dir/checkpoints/step_XXXXXX
# OR use "last" to resume from the last checkpoint:
# Format: /path/to/output_dir/checkpoints/last
CHECKPOINT_PATH="/path/to/your/checkpoint"  # <-- CHANGE THIS

# GPU device to use
CUDA_DEVICE="0"

# Optional: W&B configuration
# Uncomment to enable W&B logging
# export WANDB_API_KEY="your_wandb_api_key"
# WANDB_PROJECT="your_project"
# WANDB_ENTITY="your_entity"

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint directory does not exist: ${CHECKPOINT_PATH}"
    echo "Please set CHECKPOINT_PATH to a valid checkpoint directory."
    echo ""
    echo "Checkpoint directory should contain:"
    echo "  - config.yaml (training configuration)"
    echo "  - pretrained_model/ (model weights)"
    echo "  - training_state.pt (optimizer and scheduler state)"
    exit 1
fi

echo "==================================================================="
echo "Resuming training from checkpoint:"
echo "  ${CHECKPOINT_PATH}"
echo "==================================================================="
echo ""

# Build the command
CMD="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python src/lerobot/scripts/lerobot_train.py"
CMD="${CMD} --resume=true"
CMD="${CMD} --checkpoint_path=${CHECKPOINT_PATH}"

# Add W&B config if set
if [ ! -z "${WANDB_PROJECT}" ]; then
    CMD="${CMD} --wandb.enable=true"
    CMD="${CMD} --wandb.project=${WANDB_PROJECT}"
fi

if [ ! -z "${WANDB_ENTITY}" ]; then
    CMD="${CMD} --wandb.entity=${WANDB_ENTITY}"
fi

# Show the command being executed
echo "Executing command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}
