#!/usr/bin/env python3
"""
直接测试：Teacher用forward和sample_actions是否产生一致的结果

这个测试回答关键问题：
- Teacher用sample_actions生成X_0
- Student用forward训练
- Student用sample_actions推理
→ forward和sample_actions的计算路径是否一致？
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector, make_att_2d_masks


def test_forward_vs_sample_actions_velocity(teacher_path, device="cuda"):
    """
    关键测试：在相同的(x_t, time)下，forward计算的v_t 和
    sample_actions中denoise_step计算的v_t是否一致
    """
    print("="*80)
    print("  测试Forward vs Sample_actions的velocity一致性")
    print("="*80)

    # 加载teacher
    print(f"\n加载模型: {teacher_path}")
    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher = teacher.to(device)
    teacher.eval()

    print(f"✓ 模型加载完成")
    print(f"  max_action_dim: {teacher.config.max_action_dim}")
    print(f"  action_feature.shape[0]: {teacher.config.action_feature.shape[0]}")
    print(f"  chunk_size: {teacher.config.chunk_size}")
    print(f"  num_steps: {teacher.config.num_steps}")

    # 准备测试数据
    batch_size = 1
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim

    # 创建随机输入
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

    # 固定随机种子
    torch.manual_seed(42)
    X_1 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)
    X_0 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    # 测试在不同时间点的velocity
    test_times = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "="*80)
    print("  对比Forward和Denoise_step在不同时间点的velocity")
    print("="*80)

    max_diff_overall = 0.0

    with torch.no_grad():
        for t in test_times:
            time_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # 方法1: 使用forward计算v_t
            # forward的实现：计算 u_t = noise - actions，然后预测v_t去拟合u_t
            # 所以forward返回的是loss，我们需要直接看它内部计算的v_t

            # 我们需要手动重现forward的内部逻辑
            time_expanded = time_tensor[:, None, None]
            x_t = time_expanded * X_1 + (1 - time_expanded) * X_0
            u_t = X_1 - X_0  # 期望的velocity

            # Forward pass
            prefix_embs, prefix_pad_masks, prefix_att_masks = teacher.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks = teacher.model.embed_suffix(x_t, time_tensor)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (_, suffix_out), _ = teacher.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,  # Forward不使用cache
                fill_kv_cache=False,
            )

            suffix_out_forward = suffix_out[:, -chunk_size:]
            suffix_out_forward = suffix_out_forward.to(dtype=torch.float32)
            v_t_forward = teacher.model.action_out_proj(suffix_out_forward)

            # 方法2: 使用denoise_step计算v_t（sample_actions的方式）
            # 首先计算prefix cache
            prefix_embs, prefix_pad_masks, prefix_att_masks = teacher.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = teacher.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=teacher.config.use_cache,  # 使用cache
                fill_kv_cache=True,
            )

            # 然后调用denoise_step
            time_expanded_batch = time_tensor.expand(batch_size)
            v_t_denoise = teacher.model.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                time_expanded_batch,
            )

            # 对比差异
            diff = (v_t_forward - v_t_denoise).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            max_diff_overall = max(max_diff_overall, max_diff)

            print(f"\nt={t:.1f}:")
            print(f"  v_t_forward   : mean={v_t_forward.mean().item():.6f}, std={v_t_forward.std().item():.6f}")
            print(f"  v_t_denoise   : mean={v_t_denoise.mean().item():.6f}, std={v_t_denoise.std().item():.6f}")
            print(f"  差异 - max: {max_diff:.6e}, mean: {mean_diff:.6e}")

    print("\n" + "="*80)
    print("  总结")
    print("="*80)
    print(f"最大差异: {max_diff_overall:.6e}")

    if max_diff_overall < 1e-5:
        print("✓ Forward和Denoise_step计算的velocity完全一致！")
        print("  → 问题不在forward vs sample_actions的计算路径")
        return True
    elif max_diff_overall < 1e-3:
        print("⚠️  Forward和Denoise_step有微小差异（可能是数值精度）")
        return False
    else:
        print("✗ Forward和Denoise_step计算的velocity显著不同！")
        print("  → 这就是问题所在：训练用forward，推理用denoise_step")
        print("  → KV cache的使用导致了计算路径不一致")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    test_forward_vs_sample_actions_velocity(args.teacher_path, args.device)
