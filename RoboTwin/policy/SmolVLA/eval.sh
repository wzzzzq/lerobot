#!/bin/bash

# Set the policy name
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp
policy_name=SmolVLA

# Read command-line arguments
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mUsing policy: ${policy_name} on GPU ID: ${gpu_id}\033[0m"

# Navigate to the project root directory
#cd ../.. 

# Run the evaluation script
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}