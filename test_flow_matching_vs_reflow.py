#!/usr/bin/env python3
"""
测试Flow Matching模型和Reflow的本质区别

Flow Matching: 学习弯曲的ODE轨迹
Reflow: 学习直线轨迹

这个脚本验证：为什么用同一个模型作为teacher和student测试时loss很高
"""

import argparse
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


def test_trajectory_straightness(teacher_path: str, device: str = "cuda"):
    print("=" * 80)
    print("测试Flow Matching轨迹的弯曲程度")
    print("=" * 80)

    # Load teacher
    print(f"\n加载模型: {teacher_path}")
    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher.eval()
    teacher = teacher.to(device)

    config = teacher.config
    batch_size = 4
    chunk_size = config.chunk_size
    action_dim = config.action_feature.shape[0]
    max_action_dim = config.max_action_dim

    print(f"\n生成测试数据 (batch_size={batch_size})...")

    # Create dummy inputs
    images = torch.randn(batch_size, 1, 3, 224, 224, device=device, dtype=torch.float32)
    img_masks = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (batch_size, 20), device=device)
    lang_masks = torch.ones(batch_size, 20, device=device, dtype=torch.bool)
    state = torch.randn(batch_size, config.state_feature.shape[0], device=device, dtype=torch.float32)

    # Sample noise
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    print("\n测试1: Teacher ODE积分生成X_0")
    print("-" * 80)

    with torch.no_grad():
        X_0 = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

    print(f"X_0 shape: {X_0.shape}")
    print(f"X_0前14维 mean: {X_0[:, :, :action_dim].mean().item():.6f}")
    print(f"X_0后18维 mean: {X_0[:, :, action_dim:].mean().item():.6f}")

    print("\n测试2: 直线velocity vs ODE trajectory velocity")
    print("-" * 80)

    # 在不同时间点采样，比较直线velocity和实际预测的velocity
    times = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n对比：")
    print("  直线velocity: u_t = noise - X_0 (constant for all t)")
    print("  ODE velocity:  v_t = model预测的velocity at time t")
    print()

    # 计算直线velocity（理想的reflow target）
    u_straight = noise - X_0
    u_straight_norm = u_straight.norm(dim=-1).mean().item()

    total_deviation = 0.0

    for t_val in times:
        t = torch.tensor(t_val, dtype=torch.float32, device=device)
        t_expanded = t.expand(batch_size)

        # 构造x_t
        time_expanded = t_val  # scalar for interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * X_0

        # 预测velocity（ODE路径的velocity）
        with torch.no_grad():
            # Prepare embeddings
            prefix_embs, prefix_pad_masks, prefix_att_masks = teacher.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )

            # Compute KV cache
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = teacher.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=teacher.config.use_cache,
                fill_kv_cache=True,
            )

            # Predict velocity at this time
            v_t = teacher.model.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_expanded,
            )

        # 计算偏差
        deviation = (v_t - u_straight).norm(dim=-1).mean().item()
        deviation_pct = (deviation / u_straight_norm) * 100

        # 只看前14维
        deviation_14 = (v_t[:, :, :action_dim] - u_straight[:, :, :action_dim]).norm(dim=-1).mean().item()
        u_straight_14_norm = u_straight[:, :, :action_dim].norm(dim=-1).mean().item()
        deviation_14_pct = (deviation_14 / u_straight_14_norm) * 100

        total_deviation += deviation_14

        print(f"t={t_val}:")
        print(f"  直线velocity norm: {u_straight_14_norm:.4f} (前14维)")
        print(f"  ODE velocity偏差: {deviation_14:.4f} ({deviation_14_pct:.2f}%)")

    avg_deviation = total_deviation / len(times)
    avg_deviation_pct = (avg_deviation / u_straight_14_norm) * 100

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print(f"平均velocity偏差: {avg_deviation:.4f} ({avg_deviation_pct:.2f}%)")
    print()
    print("这解释了为什么用同一个模型测试velocity质量时loss很高：")
    print("  1. 原始模型（Flow Matching）学到的是弯曲的ODE轨迹")
    print("  2. 测试时期望的是直线velocity: u_t = noise - X_0")
    print("  3. 实际预测的是ODE轨迹的切线velocity")
    print("  4. 这两个velocity不同 → loss高")
    print()
    print("Reflow训练的目的就是：")
    print("  → 将弯曲的轨迹straighten成直线")
    print("  → 训练后的模型应该满足: v_t ≈ noise - X_0 (for all t)")
    print()
    print(f"当前模型的velocity偏差: {avg_deviation_pct:.2f}%")
    print("  → 这就是为什么需要reflow训练!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试轨迹弯曲程度")
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
    test_trajectory_straightness(args.teacher_path, args.device)
