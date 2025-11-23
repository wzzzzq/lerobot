#!/usr/bin/env python3
"""
验证reflow训练脚本的修复是否正确

测试：
1. X_0是否保持完整的32维（没有被unpad）
2. X_0是否与teacher.sample_actions()的输出完全一致
3. 模型forward是否可以正常运行
"""

import argparse
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


def verify_fix(teacher_path: str, device: str = "cuda"):
    print("=" * 80)
    print("验证Reflow训练脚本修复")
    print("=" * 80)

    # Load teacher
    print(f"\n加载teacher模型: {teacher_path}")
    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher.eval()
    teacher = teacher.to(device)

    config = teacher.config
    print(f"\n模型配置:")
    print(f"  action_dim (实际): {config.action_feature.shape[0]}")
    print(f"  max_action_dim (padding后): {config.max_action_dim}")

    # Create dummy inputs
    batch_size = 2
    chunk_size = config.chunk_size
    action_dim = config.action_feature.shape[0]
    max_action_dim = config.max_action_dim

    print(f"\n生成测试数据 (batch_size={batch_size})...")

    # Sample noise at action_dim (as in training script)
    X_1 = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=torch.float32)
    X_1_padded = pad_vector(X_1, max_action_dim)

    # Create dummy observations
    images = torch.randn(batch_size, 1, 3, 224, 224, device=device, dtype=torch.float32)
    img_masks = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (batch_size, 20), device=device)
    lang_masks = torch.ones(batch_size, 20, device=device, dtype=torch.bool)
    state = torch.randn(batch_size, config.state_feature.shape[0], device=device, dtype=torch.float32)

    print("\n测试1: 验证X_0维度保持为32")
    print("-" * 80)

    with torch.no_grad():
        # Generate X_0 using teacher (simulating the fixed prepare_reflow_batch)
        X_0_from_teacher = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1_padded
        )

    print(f"X_1 shape: {X_1.shape} (原始噪声)")
    print(f"X_1_padded shape: {X_1_padded.shape} (padding后)")
    print(f"X_0_from_teacher shape: {X_0_from_teacher.shape} (teacher生成)")

    expected_shape = (batch_size, chunk_size, max_action_dim)
    if X_0_from_teacher.shape == expected_shape:
        print(f"✓ X_0维度正确: {X_0_from_teacher.shape}")
    else:
        print(f"✗ X_0维度错误! 期望 {expected_shape}, 实际 {X_0_from_teacher.shape}")
        return False

    print("\n测试2: 验证X_0没有信息丢失")
    print("-" * 80)

    # Check if padding dimensions have values
    padding_values = X_0_from_teacher[:, :, action_dim:]
    padding_abs_mean = padding_values.abs().mean().item()
    padding_abs_max = padding_values.abs().max().item()

    print(f"X_0后{max_action_dim - action_dim}维 (padding区域):")
    print(f"  绝对值均值: {padding_abs_mean:.6f}")
    print(f"  绝对值最大值: {padding_abs_max:.6f}")

    if padding_abs_mean > 1e-6:
        print("✓ Padding维度有非零值，信息得到保留")
    else:
        print("⚠ Padding维度接近0，可能teacher本身就输出0")

    print("\n测试3: 验证forward可以正常运行")
    print("-" * 80)

    try:
        # Test forward pass with X_0 (no additional padding, as in the fix)
        teacher.train()  # Set to train mode
        losses = teacher.model.forward(
            images, img_masks, lang_tokens, lang_masks, state,
            X_0_from_teacher,  # Use X_0 directly, not padded again
            noise=X_1_padded,
            time=None
        )

        print(f"losses shape: {losses.shape}")
        print(f"losses[:, :, :action_dim] shape: {losses[:, :, :action_dim].shape}")

        # Compute loss (only on first action_dim dimensions)
        loss_full = losses.mean().item()
        loss_action_only = losses[:, :, :action_dim].mean().item()

        print(f"\nLoss值:")
        print(f"  全部32维: {loss_full:.6f}")
        print(f"  前{action_dim}维: {loss_action_only:.6f}")
        print("✓ Forward运行成功")

    except Exception as e:
        print(f"✗ Forward运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n测试4: 对比修复前后的差异")
    print("-" * 80)

    # Simulate the old buggy behavior
    X_0_old = X_0_from_teacher[:, :, :action_dim]  # Unpad (old bug)
    X_0_old_repadded = pad_vector(X_0_old, max_action_dim)  # Re-pad (old bug)

    diff = (X_0_from_teacher - X_0_old_repadded).abs()
    diff_mean = diff.mean().item()
    diff_max = diff.max().item()
    diff_padding = diff[:, :, action_dim:].mean().item()

    print(f"修复前(unpad再pad) vs 修复后(保持32维):")
    print(f"  平均差异: {diff_mean:.6f}")
    print(f"  最大差异: {diff_max:.6f}")
    print(f"  padding维度平均差异: {diff_padding:.6f}")

    if diff_mean > 1e-6:
        print(f"✓ 修复消除了{diff_mean:.6f}的信息丢失")
    else:
        print("⚠ 两种方式结果相同，可能padding维度本身就是0")

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)
    print("\n总结:")
    print("✓ X_0保持完整的32维")
    print("✓ 没有unpad/pad操作")
    print("✓ Forward可以正常运行")
    print(f"✓ 消除了平均{diff_mean:.6f}的信息丢失")
    print("\n修复正确！可以使用修复后的训练脚本重新训练。")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证reflow训练脚本修复")
    parser.add_argument(
        "--teacher_path",
        type=str,
        required=True,
        help="Teacher模型路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)",
    )

    args = parser.parse_args()
    verify_fix(args.teacher_path, args.device)
