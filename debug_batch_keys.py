#!/usr/bin/env python3
"""Debug script to check what keys are in the batch from dataset"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

# Load teacher to get config
teacher_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"
teacher = SmolVLAPolicy.from_pretrained(teacher_path)

# Create dataset config
dataset_cfg = DatasetConfig(
    repo_id="name/aloha_agix_sim",
    root="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30",
)

# Create training config
train_cfg = TrainPipelineConfig(
    dataset=dataset_cfg,
    policy=teacher.config,
    batch_size=2,
)

# Make dataset
dataset = make_dataset(train_cfg)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# Get one batch
batch = next(iter(dataloader))

print("=" * 80)
print("Batch keys:")
print("=" * 80)
for key in sorted(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
        print(f"  {key}: {batch[key].shape}")
    else:
        print(f"  {key}: {type(batch[key])}")

print("\n" + "=" * 80)
print("Checking language tokens:")
print("=" * 80)
print(f"OBS_LANGUAGE_TOKENS = '{OBS_LANGUAGE_TOKENS}'")
print(f"OBS_LANGUAGE_ATTENTION_MASK = '{OBS_LANGUAGE_ATTENTION_MASK}'")
print(f"\n'{OBS_LANGUAGE_TOKENS}' in batch: {OBS_LANGUAGE_TOKENS in batch}")
print(f"'{OBS_LANGUAGE_ATTENTION_MASK}' in batch: {OBS_LANGUAGE_ATTENTION_MASK in batch}")
