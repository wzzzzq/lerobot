#!/usr/bin/env python3
"""
对比reflow训练和robotwin eval的数据处理流程

用法:
python debug_training_vs_eval.py \
    --teacher_path /path/to/teacher/pretrained_model \
    --dataset_repo_id name/aloha_agix_sim \
    --dataset_root /path/to/dataset
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE


def print_tensor_info(name, tensor, indent=2):
    """打印tensor的详细信息"""
    prefix = " " * indent
    if isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}:")
        print(f"{prefix}  shape: {tensor.shape}")
        print(f"{prefix}  dtype: {tensor.dtype}")
        print(f"{prefix}  device: {tensor.device}")
        print(f"{prefix}  range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        # Only compute mean/std for float tensors
        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            print(f"{prefix}  mean: {tensor.mean().item():.4f}, std: {tensor.std().item():.4f}")
    elif isinstance(tensor, list):
        print(f"{prefix}{name}: list of {len(tensor)} items")
        for i, item in enumerate(tensor):
            print_tensor_info(f"[{i}]", item, indent + 2)
    else:
        print(f"{prefix}{name}: {type(tensor)}")


def simulate_training_path(teacher, batch, device):
    """模拟training时的数据处理路径"""
    print("\n" + "=" * 80)
    print("🔵 TRAINING PATH (from dataset batch)")
    print("=" * 80)

    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Training使用prepare_images和prepare_state直接处理batch
    images, img_masks = teacher.prepare_images(batch)
    state = teacher.prepare_state(batch)

    # Language tokens from batch
    if OBS_LANGUAGE_TOKENS in batch:
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    else:
        # Tokenize task field
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
        lang_tokens = tokens["input_ids"].to(device)
        lang_masks = tokens["attention_mask"].to(device).bool()

    print("\n📊 Batch keys:")
    for key in sorted(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            print(f"  {key}: {batch[key].shape}")

    print("\n📊 Raw observation.state from batch:")
    print_tensor_info("batch['observation.state']", batch[OBS_STATE])

    print("\n📊 After prepare_images:")
    for i, img in enumerate(images):
        print_tensor_info(f"images[{i}]", img)

    print("\n📊 After prepare_state:")
    print_tensor_info("state (padded)", state)

    print("\n📊 Language tokens:")
    print_tensor_info("lang_tokens", lang_tokens)
    print_tensor_info("lang_masks", lang_masks)

    return images, img_masks, lang_tokens, lang_masks, state


def simulate_eval_path(teacher, batch, device):
    """模拟eval时的数据处理路径"""
    print("\n" + "=" * 80)
    print("🟢 EVAL PATH (simulating RobotWin eval)")
    print("=" * 80)

    # 从batch提取原始数据，模拟从环境获取
    # 注意：这里假设batch的state是归一化后的，需要反归一化到原始值
    # 但实际eval从环境获取的state是什么范围？这是关键问题！

    # 1. 模拟从环境获取原始observation
    print("\n📝 Step 1: Extract 'raw' observations (simulating env.get_obs())")

    # Images: 假设环境返回uint8 [0, 255]
    # Dataset中images已经是float32 [0, 1]，需要转回uint8模拟环境
    raw_images = []
    for key in sorted(batch.keys()):
        if key.startswith("observation.images."):
            img = batch[key][0].cpu()  # 取第一个batch item
            # Convert to numpy first
            img_np = img.numpy()

            print(f"  DEBUG {key}: initial shape={img_np.shape}, dtype={img_np.dtype}")

            # 如果是[0, 1]范围，转回[0, 255]模拟环境输出
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # img_np should be (C, H, W) = (3, 480, 640)
            # CHW → HWC (环境输出格式)
            if img_np.shape[0] == 3:  # CHW format
                img_hwc = np.transpose(img_np, (1, 2, 0))
            else:  # Already HWC
                img_hwc = img_np

            raw_images.append(img_hwc)
            print(f"  {key}: shape={img_hwc.shape}, dtype={img_hwc.dtype}, range=[{img_hwc.min()}, {img_hwc.max()}]")

    # State: 假设环境返回原始joint values
    # ⚠️ 关键问题：dataset的state是否已归一化？
    raw_state = batch[OBS_STATE][0, -1 if batch[OBS_STATE].ndim > 2 else 0].cpu().numpy()
    print(f"  observation.state: shape={raw_state.shape}, dtype={raw_state.dtype}")
    print(f"    range=[{raw_state.min():.4f}, {raw_state.max():.4f}]")
    print(f"    mean={raw_state.mean():.4f}, std={raw_state.std():.4f}")
    print(f"  ⚠️  NOTE: Is this normalized or raw joint values?")

    # Task/instruction
    if "task" in batch:
        instruction = batch["task"][0] if isinstance(batch["task"], list) else batch["task"]
    else:
        instruction = "pick up the bottle"
    print(f"  task: '{instruction}'")

    # 2. 模拟eval脚本的prepare_img
    print("\n📝 Step 2: Apply eval's prepare_img (HWC→CHW, uint8→float32/255)")

    def prepare_img(img):
        # Convert HWC to CHW, normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)

    prepared_images = [prepare_img(img) for img in raw_images]
    for i, img in enumerate(prepared_images):
        print_tensor_info(f"prepared_images[{i}]", img)

    # 3. 构建observation dict (模拟eval脚本)
    print("\n📝 Step 3: Build observation dict (eval format)")

    observation = {
        "observation.state": torch.from_numpy(raw_state.astype(np.float32)),
        "task": instruction,
    }

    camera_names = ["head_camera", "left_camera", "right_camera"]
    for i, camera_name in enumerate(camera_names):
        if i < len(prepared_images):
            observation[f"observation.images.{camera_name}"] = prepared_images[i]

    print(f"  observation dict keys: {list(observation.keys())}")

    # 4. 应用preprocessor (模拟eval脚本)
    print("\n📝 Step 4: Apply preprocessor (from checkpoint)")

    # 从checkpoint加载preprocessor
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=teacher.config,
        pretrained_path=args.teacher_path,
    )

    print(f"\n  Preprocessor pipeline:")
    for step in preprocessor.steps:
        print(f"    - {step.__class__.__name__}")

    observation_window = preprocessor(observation)

    print(f"\n  Preprocessed observation_window keys: {list(observation_window.keys())}")

    # 5. 应用prepare_images和prepare_state (在_get_action_chunk中)
    print("\n📝 Step 5: Apply prepare_images and prepare_state")

    images, img_masks = teacher.prepare_images(observation_window)
    state = teacher.prepare_state(observation_window)
    lang_tokens = observation_window[OBS_LANGUAGE_TOKENS]
    lang_masks = observation_window[OBS_LANGUAGE_ATTENTION_MASK]

    print("\n📊 After prepare_images:")
    for i, img in enumerate(images):
        print_tensor_info(f"images[{i}]", img)

    print("\n📊 After prepare_state:")
    print_tensor_info("state (padded)", state)

    print("\n📊 Language tokens:")
    print_tensor_info("lang_tokens", lang_tokens)
    print_tensor_info("lang_masks", lang_masks)

    return images, img_masks, lang_tokens, lang_masks, state


def compare_inputs(train_inputs, eval_inputs):
    """对比training和eval的inputs"""
    print("\n" + "=" * 80)
    print("🔍 COMPARISON: Training vs Eval")
    print("=" * 80)

    train_images, train_img_masks, train_lang_tokens, train_lang_masks, train_state = train_inputs
    eval_images, eval_img_masks, eval_lang_tokens, eval_lang_masks, eval_state = eval_inputs

    # Compare images
    print("\n📷 Images:")
    for i in range(min(len(train_images), len(eval_images))):
        train_img = train_images[i]
        eval_img = eval_images[i]
        diff = (train_img - eval_img).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Camera {i}:")
        print(f"    Max absolute diff: {max_diff:.6f}")
        print(f"    Mean absolute diff: {mean_diff:.6f}")
        print(f"    Relative error: {mean_diff / (train_img.abs().mean().item() + 1e-8) * 100:.2f}%")

        if max_diff > 0.01:
            print(f"    ⚠️  WARNING: Large difference detected!")
            print(f"    Training: range=[{train_img.min():.4f}, {train_img.max():.4f}]")
            print(f"    Eval:     range=[{eval_img.min():.4f}, {eval_img.max():.4f}]")

    # Compare state
    print("\n🤖 State:")
    state_diff = (train_state - eval_state).abs()
    max_diff = state_diff.max().item()
    mean_diff = state_diff.mean().item()
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Relative error: {mean_diff / (train_state.abs().mean().item() + 1e-8) * 100:.2f}%")

    if max_diff > 0.01:
        print(f"  ⚠️⚠️⚠️ WARNING: Large state difference detected!")
        print(f"  Training state: range=[{train_state.min():.4f}, {train_state.max():.4f}]")
        print(f"  Eval state:     range=[{eval_state.min():.4f}, {eval_state.max():.4f}]")
        print(f"\n  This is likely the root cause of wrong actions!")
        print(f"  Possible reasons:")
        print(f"    1. Dataset state is normalized, but eval state is raw")
        print(f"    2. Preprocessor normalization not applied correctly")
        print(f"    3. Different state representations")

    # Compare language tokens
    print("\n💬 Language tokens:")
    if train_lang_tokens.shape == eval_lang_tokens.shape:
        token_match = (train_lang_tokens == eval_lang_tokens).all().item()
        print(f"  Tokens match: {token_match}")
        if not token_match:
            mismatch_count = (train_lang_tokens != eval_lang_tokens).sum().item()
            print(f"  ⚠️  Mismatch count: {mismatch_count}/{train_lang_tokens.numel()}")
    else:
        print(f"  ⚠️  Shape mismatch!")
        print(f"  Training: {train_lang_tokens.shape}")
        print(f"  Eval:     {eval_lang_tokens.shape}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Loading teacher model from: {args.teacher_path}")

    teacher = SmolVLAPolicy.from_pretrained(args.teacher_path)
    teacher.to(device)
    teacher.eval()

    print(f"✓ Model loaded")
    print(f"\n📋 Policy Config:")
    print(f"  max_action_dim: {teacher.config.max_action_dim}")
    print(f"  max_state_dim: {teacher.config.max_state_dim}")
    print(f"  action_feature.shape: {teacher.config.action_feature.shape}")
    print(f"  image_features: {teacher.config.image_features}")
    print(f"  adapt_to_pi_aloha: {teacher.config.adapt_to_pi_aloha}")
    print(f"  resize_imgs_with_padding: {teacher.config.resize_imgs_with_padding}")
    if hasattr(teacher.config, 'normalization_mapping'):
        print(f"  normalization_mapping: {teacher.config.normalization_mapping}")

    print(f"\n🗂️  Loading dataset from: {args.dataset_repo_id}")

    # Create dataset
    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=teacher.config,
        batch_size=2,
    )
    dataset = make_dataset(train_cfg)

    # Get one batch
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    batch = next(iter(dataloader))

    print(f"✓ Dataset loaded, batch size: {batch['action'].shape[0]}")

    # Simulate training path
    train_inputs = simulate_training_path(teacher, batch, device)

    # Simulate eval path
    eval_inputs = simulate_eval_path(teacher, batch, device)

    # Compare
    compare_inputs(train_inputs, eval_inputs)

    print("\n" + "=" * 80)
    print("✅ Analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比training和eval的数据处理")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to teacher model checkpoint")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="Dataset repository ID")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()
    main(args)
