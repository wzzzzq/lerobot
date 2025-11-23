#!/usr/bin/env python3
"""
验证Student的velocity预测质量

测试：
1. 使用真实dataset的观测数据
2. Teacher生成X_0（从噪声）
3. 在随机时间点t，计算真实的u_t = X_1 - X_0
4. 让Student预测v_t
5. 对比误差：这应该接近训练loss

如果这个误差很大（>10%），说明Student没学好
如果这个误差小但积分结果差，说明ODE积分有问题
"""

import torch
import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig


def test_student_velocity_quality(
    teacher_path,
    student_path,
    dataset_repo_id=None,
    dataset_root=None,
    device="cuda"
):
    """测试Student预测velocity的质量"""
    print("="*80)
    print("  测试Student的Velocity预测质量")
    print("="*80)

    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    student = SmolVLAPolicy.from_pretrained(student_path)

    teacher = teacher.to(device)
    student = student.to(device)

    teacher.eval()
    student.eval()

    # 准备测试数据
    batch_size = 32
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim
    action_dim = teacher.config.action_feature.shape[0]

    # Load real dataset if provided
    use_real_data = dataset_repo_id is not None

    if use_real_data:
        print(f"\n✓ 使用真实dataset: {dataset_repo_id}")

        dataset_cfg = DatasetConfig(
            repo_id=dataset_repo_id,
            root=dataset_root,
        )

        # Create minimal training config
        train_cfg = TrainPipelineConfig(
            dataset=dataset_cfg,
            policy=teacher.config,
            batch_size=batch_size,
        )

        dataset = make_dataset(train_cfg)

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Get one batch
        batch = next(iter(dataloader))

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Prepare observations using policy's methods
        images, img_masks = teacher.prepare_images(batch)
        state = teacher.prepare_state(batch)

        # Try to get language tokens from batch, fallback to default if not present
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

        if f"{OBS_LANGUAGE_TOKENS}" in batch:
            lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
            lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            print(f"  Images: {len(images)} cameras")
            print(f"  State shape: {state.shape}")
            print(f"  Language tokens shape: {lang_tokens.shape}")
        else:
            # Dataset doesn't have language annotations, use default text
            print(f"  ⚠️  Dataset没有language annotations，使用默认文本")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
            tokens = tokenizer("pick up the bottle", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
            lang_tokens = tokens["input_ids"].to(device).repeat(batch_size, 1)
            lang_masks = tokens["attention_mask"].to(device).bool().repeat(batch_size, 1)
            print(f"  Images: {len(images)} cameras")
            print(f"  State shape: {state.shape}")
            print(f"  Language tokens: 使用默认 'pick up the bottle'")
    else:
        print("\n⚠️  警告: 未提供dataset，使用随机数据（结果不可靠）")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
        tokens = tokenizer("pick up the bottle", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
        lang_tokens = tokens["input_ids"].to(device).repeat(batch_size, 1)
        lang_masks = tokens["attention_mask"].to(device).bool().repeat(batch_size, 1)

        img_size = teacher.config.resize_imgs_with_padding if teacher.config.resize_imgs_with_padding else (512, 512)
        images = [torch.randn(batch_size, 3, img_size[0], img_size[1], device=device) for _ in range(3)]
        img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(3)]
        state = torch.randn(batch_size, teacher.config.max_state_dim, device=device)

    print("\n" + "="*80)
    print("  测试Student在不同时间点预测velocity的误差")
    print("="*80)

    # CRITICAL: Match training noise distribution
    # Training uses: 14-dim random + 18-dim zeros
    torch.manual_seed(42)
    X_1 = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=torch.float32)  # 14-dim
    X_1_padded = pad_vector(X_1, max_action_dim)  # Pad to 32-dim with zeros

    print(f"\n噪声分布 (匹配训练):")
    print(f"  X_1: {X_1.shape} (前{action_dim}维)")
    print(f"  X_1_padded: {X_1_padded.shape} (后{max_action_dim - action_dim}维=0)")

    with torch.no_grad():
        # Teacher生成X_0
        X_0 = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1_padded.clone()
        )

        print(f"  X_0: {X_0.shape} (teacher生成)")

        # 真实的straight-line velocity
        u_true = X_1_padded - X_0  # shape: (batch, chunk, max_action_dim)

        print(f"\n真实velocity统计:")
        print(f"  全部{max_action_dim}维: mean={u_true.mean().item():.6f}, std={u_true.std().item():.6f}")
        print(f"  前{action_dim}维: mean={u_true[:,:,:action_dim].mean().item():.6f}")

        # 测试多个时间点
        test_times = [0.1, 0.3, 0.5, 0.7, 0.9]
        errors = []

        for t in test_times:
            time_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # Student预测velocity
            losses = student.model.forward(
                images, img_masks, lang_tokens, lang_masks, state,
                X_0, noise=X_1_padded, time=time_tensor
            )

            # 只看有效维度（前14维）
            losses_valid = losses[:, :, :action_dim]
            mse_loss_valid = losses_valid.mean().item()
            rmse_valid = np.sqrt(mse_loss_valid)

            # 计算相对误差
            u_scale = u_true[:,:,:action_dim].abs().mean().item()
            relative_error = rmse_valid / (u_scale + 1e-8) * 100

            errors.append({
                't': t,
                'mse': mse_loss_valid,
                'rmse': rmse_valid,
                'relative': relative_error
            })

            print(f"\nt={t:.1f}:")
            print(f"  MSE loss (前{action_dim}维): {mse_loss_valid:.6f}")
            print(f"  RMSE: {rmse_valid:.6f}")
            print(f"  相对误差: {relative_error:.2f}%")

    print("\n" + "="*80)
    print("  总结")
    print("="*80)

    avg_mse = np.mean([e['mse'] for e in errors])
    avg_rmse = np.mean([e['rmse'] for e in errors])
    avg_relative = np.mean([e['relative'] for e in errors])

    print(f"平均MSE loss: {avg_mse:.6f}")
    print(f"平均RMSE: {avg_rmse:.6f}")
    print(f"平均相对误差: {avg_relative:.2f}%")

    print(f"\n训练报告的loss: ~0.002")
    print(f"这个测试的平均loss: {avg_mse:.6f}")
    print(f"数据来源: {'真实dataset' if use_real_data else '随机数据（不可靠）'}")

    if not use_real_data:
        print("\n⚠️  注意: 使用随机数据测试，结果仅供参考！")
        print("   建议使用 --dataset_repo_id 和 --dataset_root 参数提供真实数据")
    elif avg_mse > 0.005:
        print("\n⚠️  测试loss比训练loss大！")
        print(f"  差异: {avg_mse / 0.002:.1f}x")
        if avg_mse > 0.05:
            print("  → 可能还在训练中，或训练有问题")
        else:
            print("  → 轻微差异是正常的（测试batch vs 训练平均）")
    elif avg_relative > 15:
        print(f"\n✗ 平均相对误差{avg_relative:.1f}%太大！")
        print("  → Student的预测精度不够")
        print("  → 10步ODE积分会放大误差")
    elif avg_relative > 5:
        print(f"\n⚠️  平均相对误差{avg_relative:.1f}%偏大")
        print("  → 10步ODE积分后可能达到20-30%")
    else:
        print(f"\n✓ 平均相对误差{avg_relative:.1f}%良好")
        print("  → Velocity预测质量符合预期")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试Student velocity质量（推荐使用真实dataset）")
    parser.add_argument("--teacher_path", type=str, required=True, help="Teacher模型路径")
    parser.add_argument("--student_path", type=str, required=True, help="Student模型路径")
    parser.add_argument("--dataset_repo_id", type=str, default=None, help="Dataset repo ID (推荐)")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    test_student_velocity_quality(
        args.teacher_path,
        args.student_path,
        args.dataset_repo_id,
        args.dataset_root,
        args.device
    )
