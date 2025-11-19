#!/bin/bash

# Evaluation script for Put Bottles Dustbin task with SmolVLA
#
# This script evaluates a trained SmolVLA policy on the RoboTwin put_bottles_dustbin task.
# It creates a temporary config file with all settings and runs the evaluation.
#
# Usage:
#   bash examples/robotwin_eval/eval_put_bottles_dustbin.sh

# ==========================
# Configuration
# ==========================

# Policy settings
POLICY_PATH="/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow/checkpoints"
CKPT_SETTING="last"  # Options: "last", "best", or specific checkpoint path

# Task setting
TASK_NAME="put_bottles_dustbin"
TASK_CONFIG="demo_clean"  # Options: "demo_clean", "randomized", etc.

# Evaluation settings
NUM_EPISODES=3 # Number of episodes to evaluate
SEED=42
INSTRUCTION_TYPE="seen"  # Options: "seen", "unseen"

# NOTE: These settings MUST match training config (from config.json):
# - n_action_steps: 50 (chunk_size from training)
# - tokenizer_max_length: 48 (from training config)
N_ACTION_STEPS=50
TOKENIZER_MAX_LENGTH=48

# Denoising steps (optional - leave empty to use checkpoint's default)
NUM_STEPS="10"  # Empty = use checkpoint default
# NUM_STEPS=2  # Uncomment and set for 2-RF models

# GPU settings
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Environment variables (optional)
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp
export http_proxy=http://172.16.0.136:18000
export https_proxy=http://172.16.0.136:18000

# ==========================
# Script Start
# ==========================

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m  SmolVLA Evaluation - Put Bottles Dustbin\033[0m"
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
n_action_steps: ${N_ACTION_STEPS}

# Model settings (MUST match training config)
policy_name: SmolVLA
tokenizer_max_length: ${TOKENIZER_MAX_LENGTH}
EOF

# Add num_steps if specified
if [ -n "${NUM_STEPS}" ]; then
    echo "num_steps: ${NUM_STEPS}" >> ${CONFIG_FILE}
    echo -e "\033[33mDenoising steps: ${NUM_STEPS}\033[0m"
fi

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
