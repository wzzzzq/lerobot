#!/usr/bin/env python3
"""检查teacher和student的preprocessing是否一致"""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

teacher_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"
student_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow_new/checkpoints/last/pretrained_model"
dataset_repo_id = "name/aloha_agix_sim"
dataset_root = "/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"

print("Loading models...")
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
student = SmolVLAPolicy.from_pretrained(student_path)
teacher.eval()
student.eval()

print("Loading dataset...")
dataset_cfg = DatasetConfig(repo_id=dataset_repo_id, root=dataset_root)
train_cfg = TrainPipelineConfig(dataset=dataset_cfg, policy=teacher.config, batch_size=2)
dataset = make_dataset(train_cfg)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(dataloader))

# Move to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(device)

print('\n' + '=' * 80)
print('PREPROCESSING CONFIG COMPARISON')
print('=' * 80)

print('\n📋 Image preprocessing config:')
print(f'  Teacher resize_imgs_with_padding: {teacher.config.resize_imgs_with_padding}')
print(f'  Student resize_imgs_with_padding: {student.config.resize_imgs_with_padding}')
if teacher.config.resize_imgs_with_padding != student.config.resize_imgs_with_padding:
    print('  ⚠️⚠️⚠️ MISMATCH!')

print('\n📋 State/action config:')
print(f'  Teacher max_state_dim: {teacher.config.max_state_dim}')
print(f'  Student max_state_dim: {student.config.max_state_dim}')
if teacher.config.max_state_dim != student.config.max_state_dim:
    print('  ⚠️⚠️⚠️ MISMATCH!')

print(f'  Teacher max_action_dim: {teacher.config.max_action_dim}')
print(f'  Student max_action_dim: {student.config.max_action_dim}')
if teacher.config.max_action_dim != student.config.max_action_dim:
    print('  ⚠️⚠️⚠️ MISMATCH!')

print('\n' + '=' * 80)
print('ACTUAL PREPROCESSING OUTPUT COMPARISON')
print('=' * 80)

print('\n🔵 Using teacher.prepare_images/prepare_state:')
with torch.no_grad():
    teacher_images, teacher_img_masks = teacher.prepare_images(batch)
    teacher_state = teacher.prepare_state(batch)

print(f'  Images: {len(teacher_images)} cameras')
for i, img in enumerate(teacher_images):
    print(f'    Camera {i}: shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}]')
print(f'  State: shape={teacher_state.shape}, range=[{teacher_state.min():.4f}, {teacher_state.max():.4f}]')

print('\n🟢 Using student.prepare_images/prepare_state:')
with torch.no_grad():
    student_images, student_img_masks = student.prepare_images(batch)
    student_state = student.prepare_state(batch)

print(f'  Images: {len(student_images)} cameras')
for i, img in enumerate(student_images):
    print(f'    Camera {i}: shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}]')
print(f'  State: shape={student_state.shape}, range=[{student_state.min():.4f}, {student_state.max():.4f}]')

print('\n🔍 Differences:')
all_match = True
for i in range(min(len(teacher_images), len(student_images))):
    diff = (teacher_images[i] - student_images[i]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f'  Camera {i}:')
    print(f'    Max diff: {max_diff:.6f}')
    print(f'    Mean diff: {mean_diff:.6f}')
    if max_diff > 1e-5:
        print(f'    ⚠️⚠️⚠️ IMAGES DIFFER!')
        all_match = False

state_diff = (teacher_state - student_state).abs()
max_diff = state_diff.max().item()
mean_diff = state_diff.mean().item()
print(f'  State:')
print(f'    Max diff: {max_diff:.6f}')
print(f'    Mean diff: {mean_diff:.6f}')
if max_diff > 1e-5:
    print(f'    ⚠️⚠️⚠️ STATE DIFFERS!')
    all_match = False

if all_match:
    print('\n✓ All preprocessing outputs match!')
else:
    print('\n⚠️  PREPROCESSING MISMATCH DETECTED!')
    print('This means teacher and student see different observations during reflow training!')
