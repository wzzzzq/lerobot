#!/bin/bash

# Evaluation script for Stack Blocks task with SmolVLA
#
# This script evaluates a trained SmolVLA policy on the RoboTwin stack_blocks_three task.
#
# Usage:
#   bash examples/robotwin_eval/eval_stack_blocks.sh

# ==========================
# Configuration
# ==========================

# Policy settings
POLICY_PATH="/path/to/your/stack_blocks/checkpoints"
CKPT_SETTING="last"  # Options: "last", "best", or specific checkpoint path

# Task settings
TASK_NAME="stack_blocks_three"
TASK_CONFIG="demo_clean"  # Options: "demo_clean", "randomized", etc.

# Evaluation settings
NUM_EPISODES=10  # Number of episodes to evaluate
SEED=42
INSTRUCTION_TYPE="seen"  # Options: "seen", "unseen"

# GPU settings
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ==========================
# Script Start
# ==========================

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m  SmolVLA Evaluation - Stack Blocks\033[0m"
echo -e "\033[33m========================================\033[0m"
echo -e "\033[33mPolicy Path: ${POLICY_PATH}\033[0m"
echo -e "\033[33mCheckpoint: ${CKPT_SETTING}\033[0m"
echo -e "\033[33mTask Config: ${TASK_CONFIG}\033[0m"
echo -e "\033[33mGPU: ${CUDA_VISIBLE_DEVICES}\033[0m"
echo -e "\033[33mSeed: ${SEED}\033[0m"
echo -e "\033[33mNum Episodes: ${NUM_EPISODES}\033[0m"
echo -e "\033[33m========================================\033[0m"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LEROBOT_ROOT="$(dirname $(dirname ${SCRIPT_DIR}))"

# Change to RoboTwin directory for evaluation
cd "${LEROBOT_ROOT}/RoboTwin" || {
    echo "Error: RoboTwin directory not found at ${LEROBOT_ROOT}/RoboTwin"
    exit 1
}

# Create temporary config file
CONFIG_FILE="/tmp/eval_config_${TASK_NAME}_$$.yml"

cat > ${CONFIG_FILE} <<EOF
# SmolVLA Evaluation Config for ${TASK_NAME}
# Generated at: $(date)

# Policy configuration
policy_path: ${POLICY_PATH}
ckpt_setting: "${CKPT_SETTING}"
device: cuda

# Task configuration
task_name: ${TASK_NAME}
task_config: ${TASK_CONFIG}
instruction_type: ${INSTRUCTION_TYPE}

# Evaluation settings
num_episodes: ${NUM_EPISODES}
seed: ${SEED}
n_action_steps: 10

# Model settings (for compatibility)
policy_name: SmolVLA
tokenizer_max_length: 96
EOF

echo "Config file created: ${CONFIG_FILE}"
echo ""

# Run evaluation
python ../examples/robotwin_eval/eval_policy_smolvla.py --config ${CONFIG_FILE}

# Capture exit code
EXIT_CODE=$?

# Cleanup
rm -f ${CONFIG_FILE}

if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "\n\033[32m✓ Evaluation completed successfully!\033[0m"
else
    echo -e "\n\033[31m✗ Evaluation failed with exit code ${EXIT_CODE}\033[0m"
fi

exit ${EXIT_CODE}
