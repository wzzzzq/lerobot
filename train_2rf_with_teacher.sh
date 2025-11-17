#!/bin/bash
# Train 2-RF with teacher model using Reflow algorithm
# Teacher model: /pfs/pfs-ilWc5D/ziqianwang/pretrain_put_bottles_dustbin/checkpoints
# Based on REFLOW_SMOLVLA_GUIDE.md Algorithm 1

export http_proxy=http://172.16.0.136:18000 
export https_proxy=http://172.16.0.136:18000
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp

export WANDB_API_KEY="489fe7b734df1e91930d434d63c36b600b2faed9"

# RTX 4000 series GPU support
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Generate unique run_id with timestamp
RUN_ID="2rf_put_bottles_dustbin_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=3 python src/lerobot/scripts/lerobot_train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path=/pfs/pfs-ilWc5D/ziqianwang/pretrain_put_bottles_dustbin/checkpoints/020000/pretrained_model \
  --dataset.root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30 \
  --dataset.repo_id=name/aloha_agix_sim \
  --policy.tokenizer_max_length=96 \
  --batch_size=32 \
  --steps=50000 \
  --policy.device=cuda \
  --output_dir=/pfs/pfs-ilWc5D/ziqianwang/2rf_put_bottles_dustbin \
  --wandb.enable=true \
  --wandb.project=aloha_smolvla \
  --wandb.entity=christianwang-sjtu \
  --wandb.run_id=${RUN_ID} \
  --wandb.mode=online \
  --policy.push_to_hub=false

# Notes:
# - When use_reflow=true, student model automatically inherits teacher weights
# - VLM is frozen, only action expert layers are fine-tuned
# - No need to specify --policy.pretrained_path (handled automatically)
# - Changed output_dir to 2rf_put_bottles_dustbin to distinguish from pretrain
# - Changed run_id prefix to 2rf_put_bottles_dustbin for W&B tracking
