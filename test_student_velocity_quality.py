#!/usr/bin/env python3
"""
验证Student的velocity预测质量

测试：
1. Teacher生成X_0（从噪声）
2. 在随机时间点t，计算真实的u_t = X_1 - X_0
3. 让Student预测v_t
4. 对比误差：这应该接近训练loss

如果这个误差很大（>10%），说明Student没学好
如果这个误差小但积分结果差，说明ODE积分有问题
"""

import torch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def test_student_velocity_quality(teacher_path, student_path, device="cuda"):
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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
    tokens = tokenizer("pick up the bottle", return_tensors="pt", padding="max_length", max_length=48, truncation=True)
    lang_tokens = tokens["input_ids"].to(device)
    lang_masks = tokens["attention_mask"].to(device).bool()  # Ensure boolean type

    # Use correct image size (512x512 for SmolVLA)
    img_size = teacher.config.resize_imgs_with_padding if teacher.config.resize_imgs_with_padding else (512, 512)
    images = [torch.randn(batch_size, 3, img_size[0], img_size[1], device=device) for _ in range(3)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(3)]
    state = torch.randn(batch_size, teacher.config.max_state_dim, device=device)

    print("\n" + "="*80)
    print("  测试Student在不同时间点预测velocity的误差")
    print("="*80)

    torch.manual_seed(42)
    X_1 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Teacher生成X_0
        X_0 = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1.clone()
        )

        # 真实的straight-line velocity
        u_true = X_1 - X_0  # shape: (batch, chunk, max_action_dim)

        print(f"\n真实velocity统计:")
        print(f"  mean={u_true.mean().item():.6f}, std={u_true.std().item():.6f}")
        print(f"  前{action_dim}维 mean={u_true[:,:,:action_dim].mean().item():.6f}")

        # 测试多个时间点
        test_times = [0.1, 0.3, 0.5, 0.7, 0.9]
        errors = []

        for t in test_times:
            time_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # 计算x_t
            time_expanded = time_tensor[:, None, None]
            x_t = time_expanded * X_1 + (1 - time_expanded) * X_0

            # Student预测velocity
            # 方法1: 通过forward获取loss（间接）
            losses = student.model.forward(
                images, img_masks, lang_tokens, lang_masks, state,
                X_0, noise=X_1, time=time_tensor
            )

            # Loss是MSE(u_true, v_pred)
            mse_loss = losses.mean().item()
            rmse = np.sqrt(mse_loss)

            # 只看有效维度
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

    if avg_mse > 0.005:
        print("\n⚠️  测试loss比训练loss大很多！")
        print("  → 可能是过拟合，或测试数据分布不同")
    elif avg_relative > 15:
        print(f"\n✗ 平均相对误差{avg_relative:.1f}%太大！")
        print("  → Student的预测精度不够")
        print("  → 10步ODE积分会放大误差到27.93%")
        print("\n建议:")
        print("  1. 增加训练步数（当前40000步可能不够）")
        print("  2. 提高学习率（当前2.5e-6可能太小）")
        print("  3. 检查训练数据质量")
    elif avg_relative > 5:
        print(f"\n⚠️  平均相对误差{avg_relative:.1f}%偏大")
        print("  → 虽然不算很大，但ODE积分会累积")
        print("  → 10步后可能达到20-30%")
    else:
        print(f"\n✓ 平均相对误差{avg_relative:.1f}%可接受")
        print("  → 问题可能在ODE积分的其他方面")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    test_student_velocity_quality(args.teacher_path, args.student_path, args.device)
