#!/usr/bin/env python3
"""
对比teacher和student在相同输入下的action输出

这个测试模拟eval环境，检查：
1. Teacher和student在相同observation下生成的actions是否接近
2. 如果差异很大，说明student训练有问题
"""
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

teacher_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"
student_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow_new/checkpoints/last/pretrained_model"
dataset_repo_id = "name/aloha_agix_sim"
dataset_root = "/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading models...")
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
student = SmolVLAPolicy.from_pretrained(student_path)
teacher.to(device)
student.to(device)
teacher.eval()
student.eval()

print("Loading dataset...")
dataset_cfg = DatasetConfig(repo_id=dataset_repo_id, root=dataset_root)
train_cfg = TrainPipelineConfig(dataset=dataset_cfg, policy=teacher.config, batch_size=1)
dataset = make_dataset(train_cfg)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(dataloader))

# Move to device
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(device)

print('\n' + '=' * 80)
print('COMPARING TEACHER vs STUDENT ACTIONS')
print('=' * 80)

print("\n📊 Batch keys:")
for key in sorted(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
        print(f"  {key}: {batch[key].shape}")

# Generate actions using SAME noise for both models
print("\n🎲 Creating shared random noise...")
action_dim = teacher.config.action_feature.shape[0]
max_action_dim = teacher.config.max_action_dim
n_action_steps = teacher.config.n_action_steps

# Sample noise at action_dim (14)
X_1 = torch.randn(1, n_action_steps, action_dim, device=device, dtype=torch.float32)

# Pad to max_action_dim (32) for model input
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
X_1_padded = pad_vector(X_1, max_action_dim)

print(f"  Noise shape: {X_1_padded.shape}")
print(f"  Noise stats: mean={X_1_padded.mean():.4f}, std={X_1_padded.std():.4f}")

# ============================================================================
# TEACHER: Completely independent processing
# ============================================================================
print("\n" + "=" * 80)
print("🔵 TEACHER PROCESSING (completely independent)")
print("=" * 80)

# Teacher: prepare inputs independently
teacher_images, teacher_img_masks = teacher.prepare_images(batch)
teacher_state = teacher.prepare_state(batch)

# Teacher: tokenize language independently
if OBS_LANGUAGE_TOKENS in batch:
    teacher_lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    teacher_lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
    task = batch["task"]
    task_texts = [task] if isinstance(task, str) else task
    task_texts = [t if t.endswith("\n") else f"{t}\n" for t in task_texts]
    tokens = tokenizer(
        task_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=teacher.config.tokenizer_max_length,
        truncation=True
    )
    teacher_lang_tokens = tokens["input_ids"].to(device)
    teacher_lang_masks = tokens["attention_mask"].to(device).bool()

print(f"  Images: {len(teacher_images)} cameras, shape={teacher_images[0].shape}")
print(f"  State: {teacher_state.shape}, mean={teacher_state.mean():.4f}")
print(f"  Language tokens: {teacher_lang_tokens.shape}")

# Teacher: generate actions
print("\n  Generating actions...")
with torch.no_grad():
    teacher_actions_full = teacher.model.sample_actions(
        teacher_images, teacher_img_masks, teacher_lang_tokens, teacher_lang_masks,
        teacher_state, noise=X_1_padded.clone()  # Clone noise to be safe
    )
    # Unpad to original action_dim (as done in inference)
    teacher_actions = teacher_actions_full[:, :, :action_dim]

print(f"  Actions shape: {teacher_actions.shape}")
print(f"  Actions range: [{teacher_actions.min():.4f}, {teacher_actions.max():.4f}]")
print(f"  Actions mean: {teacher_actions.mean():.4f}, std: {teacher_actions.std():.4f}")

# ============================================================================
# STUDENT: Completely independent processing
# ============================================================================
print("\n" + "=" * 80)
print("🟢 STUDENT PROCESSING (completely independent)")
print("=" * 80)

# Student: prepare inputs independently
student_images, student_img_masks = student.prepare_images(batch)
student_state = student.prepare_state(batch)

# Student: tokenize language independently
if OBS_LANGUAGE_TOKENS in batch:
    student_lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    student_lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student.config.vlm_model_name)
    task = batch["task"]
    task_texts = [task] if isinstance(task, str) else task
    task_texts = [t if t.endswith("\n") else f"{t}\n" for t in task_texts]
    tokens = tokenizer(
        task_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=student.config.tokenizer_max_length,
        truncation=True
    )
    student_lang_tokens = tokens["input_ids"].to(device)
    student_lang_masks = tokens["attention_mask"].to(device).bool()

print(f"  Images: {len(student_images)} cameras, shape={student_images[0].shape}")
print(f"  State: {student_state.shape}, mean={student_state.mean():.4f}")
print(f"  Language tokens: {student_lang_tokens.shape}")

# Student: generate actions
print("\n  Generating actions...")
with torch.no_grad():
    student_actions_full = student.model.sample_actions(
        student_images, student_img_masks, student_lang_tokens, student_lang_masks,
        student_state, noise=X_1_padded.clone()  # Clone noise to be safe
    )
    # Unpad to original action_dim
    student_actions = student_actions_full[:, :, :action_dim]

print(f"  Actions shape: {student_actions.shape}")
print(f"  Actions range: [{student_actions.min():.4f}, {student_actions.max():.4f}]")
print(f"  Actions mean: {student_actions.mean():.4f}, std: {student_actions.std():.4f}")

# Compare actions
print("\n🔍 COMPARISON:")
diff = (teacher_actions - student_actions).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
relative_err = mean_diff / (teacher_actions.abs().mean().item() + 1e-8) * 100

print(f"  Max absolute difference: {max_diff:.6f}")
print(f"  Mean absolute difference: {mean_diff:.6f}")
print(f"  Relative error: {relative_err:.2f}%")

# Per-dimension comparison
print("\n📊 Per-dimension comparison (first 14 dims):")
for i in range(min(14, action_dim)):
    t_val = teacher_actions[0, 0, i].item()
    s_val = student_actions[0, 0, i].item()
    diff_val = abs(t_val - s_val)
    print(f"  Dim {i:2d}: teacher={t_val:8.4f}, student={s_val:8.4f}, diff={diff_val:.6f}")

print("\n" + "=" * 80)
print("VERDICT:")
print("=" * 80)

if relative_err < 5:
    print("✅ Student actions are VERY CLOSE to teacher (< 5% error)")
    print("   Problem is likely NOT in the student model itself")
    print("   Check eval script or environment setup")
elif relative_err < 20:
    print("⚠️  Student actions have MODERATE difference (5-20% error)")
    print("   This might accumulate during rollout")
    print("   Student may need more training or better hyperparameters")
else:
    print("🔴 Student actions are VERY DIFFERENT from teacher (> 20% error)")
    print("   Student training has FAILED!")
    print("   Possible causes:")
    print("   1. Student was trained with buggy code (unpad/pad bug)")
    print("   2. Training didn't converge (check loss curve)")
    print("   3. Wrong training hyperparameters")
    print("   4. Model weights corrupted")
    print("\n   SOLUTION: Retrain student with fixed code")

# Also check the padding dimensions
print("\n" + "=" * 80)
print("CHECKING PADDING DIMENSIONS (14-32):")
print("=" * 80)
print(f"Teacher padding dims: range=[{teacher_actions_full[0, 0, action_dim:].min():.6f}, {teacher_actions_full[0, 0, action_dim:].max():.6f}], mean={teacher_actions_full[0, 0, action_dim:].abs().mean():.6f}")
print(f"Student padding dims: range=[{student_actions_full[0, 0, action_dim:].min():.6f}, {student_actions_full[0, 0, action_dim:].max():.6f}], mean={student_actions_full[0, 0, action_dim:].abs().mean():.6f}")

if student_actions_full[0, 0, action_dim:].abs().mean() > 0.01:
    print("⚠️  Student has non-zero padding dims! This is expected after reflow training.")
else:
    print("✓ Student padding dims are close to zero")
