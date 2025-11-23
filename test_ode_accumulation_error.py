#!/usr/bin/env python3
"""
验证ODE积分的累积误差：forward vs sample_actions

测试：从相同噪声开始，用两种方法积分到t=0，对比最终的X_0差异
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks


def test_ode_accumulation_error(teacher_path, device="cuda"):
    """
    测试累积误差：
    1. 用sample_actions积分10步得到X_0_sample
    2. 用forward模拟积分10步得到X_0_forward
    3. 对比差异
    """
    print("="*80)
    print("  测试ODE积分累积误差")
    print("="*80)

    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher = teacher.to(device)
    teacher.eval()

    print(f"\n✓ 模型加载完成")
    print(f"  num_steps: {teacher.config.num_steps}")

    # 准备测试数据
    batch_size = 1
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
    tokens = tokenizer("pick up the bottle", return_tensors="pt", padding=True, max_length=48)
    lang_tokens = tokens["input_ids"].to(device)
    lang_masks = tokens["attention_mask"].to(device)

    images = [torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(3)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(3)]
    state = torch.randn(batch_size, teacher.config.max_state_dim, device=device)

    # 固定随机种子
    torch.manual_seed(42)
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    print("\n" + "="*80)
    print("  方法1: 用sample_actions积分（正确的推理路径）")
    print("="*80)

    with torch.no_grad():
        X_0_sample = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )

    print(f"X_0_sample: mean={X_0_sample.mean().item():.6f}, std={X_0_sample.std().item():.6f}")

    print("\n" + "="*80)
    print("  方法2: 手动模拟forward路径积分（训练路径）")
    print("="*80)

    with torch.no_grad():
        # 手动Euler积分，但每步用forward计算velocity
        dt = -1.0 / teacher.config.num_steps
        x_t = noise.clone()
        time = 1.0

        step = 0
        while time >= -dt / 2:
            time_tensor = torch.tensor([time], device=device, dtype=torch.float32)

            # 用forward的方式计算velocity
            time_expanded = time_tensor[:, None, None]

            # Forward路径：不使用KV cache
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
                use_cache=False,
                fill_kv_cache=False,
            )

            suffix_out_forward = suffix_out[:, -chunk_size:]
            suffix_out_forward = suffix_out_forward.to(dtype=torch.float32)
            v_t = teacher.model.action_out_proj(suffix_out_forward)

            # Euler步
            x_t = x_t + dt * v_t
            time = time + dt
            step += 1

            if step <= 3 or step == teacher.config.num_steps:
                print(f"  Step {step}: time={time:.2f}, x_t mean={x_t.mean().item():.6f}")

        X_0_forward = x_t

    print(f"\nX_0_forward: mean={X_0_forward.mean().item():.6f}, std={X_0_forward.std().item():.6f}")

    print("\n" + "="*80)
    print("  对比差异")
    print("="*80)

    diff = (X_0_sample - X_0_forward).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    scale = X_0_sample.abs().mean().item()
    relative_diff = mean_diff / (scale + 1e-8) * 100

    print(f"绝对差异:")
    print(f"  最大: {max_diff:.6f}")
    print(f"  平均: {mean_diff:.6f}")
    print(f"相对差异: {relative_diff:.2f}%")

    print("\n" + "="*80)
    print("  结论")
    print("="*80)

    if relative_diff < 1.0:
        print("✓ 累积误差很小（<1%），不是主要问题")
    elif relative_diff < 10.0:
        print("⚠️  累积误差中等（1-10%），可能是问题的一部分")
    else:
        print("✗ 累积误差很大（>10%），这就是主要问题！")
        print("  → Student用forward训练，但用sample_actions推理")
        print("  → 10步积分的累积误差导致action完全不对")

    return relative_diff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    test_ode_accumulation_error(args.teacher_path, args.device)
