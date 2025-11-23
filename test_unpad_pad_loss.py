#!/usr/bin/env python3
"""
验证X_0 unpad/pad的信息损失问题

测试：
1. Teacher从X_1_padded(32维)积分得到X_0_full(32维)
2. Unpad到X_0_14(14维)，再pad回X_0_padded(32维，后18维补0)
3. 检查X_0_full和X_0_padded是否一致

如果不一致，说明unpad/pad过程丢失了信息！
这会导致训练时的target(X_0_padded)和teacher实际生成的(X_0_full)不同！
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


def test_unpad_pad_information_loss(teacher_path, device="cuda"):
    """测试unpad/pad是否导致信息损失"""
    print("="*80)
    print("  测试X_0 Unpad/Pad的信息损失")
    print("="*80)

    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher = teacher.to(device)
    teacher.eval()

    batch_size = 1
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim
    action_dim = teacher.config.action_feature.shape[0]

    print(f"\n模型配置:")
    print(f"  action_dim (实际): {action_dim}")
    print(f"  max_action_dim (padding后): {max_action_dim}")
    print(f"  padding维度: {max_action_dim - action_dim}")

    # 准备测试数据
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

    # 测试：从padding的噪声生成X_0
    torch.manual_seed(42)
    X_1 = torch.randn(batch_size, chunk_size, action_dim, device=device, dtype=torch.float32)
    X_1_padded = pad_vector(X_1, max_action_dim)

    print(f"\n噪声X_1:")
    print(f"  前{action_dim}维: mean={X_1.mean().item():.6f}, std={X_1.std().item():.6f}")
    print(f"  后{max_action_dim-action_dim}维: 全部为0 (padding)")

    with torch.no_grad():
        # Teacher生成完整的X_0 (32维)
        X_0_full = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1_padded
        )

        print(f"\nTeacher生成的X_0_full (32维):")
        print(f"  前{action_dim}维: mean={X_0_full[:,:,:action_dim].mean().item():.6f}")
        print(f"  后{max_action_dim-action_dim}维: mean={X_0_full[:,:,action_dim:].mean().item():.6f}, "
              f"std={X_0_full[:,:,action_dim:].std().item():.6f}")

        # 检查后18维是否全为0
        padding_dims_mean = X_0_full[:,:,action_dim:].abs().mean().item()
        padding_dims_max = X_0_full[:,:,action_dim:].abs().max().item()

        print(f"  后18维绝对值: mean={padding_dims_mean:.6f}, max={padding_dims_max:.6f}")

        # 模拟训练脚本的unpad/pad操作
        X_0_unpadded = X_0_full[:, :, :action_dim]  # Unpad到14维
        X_0_repadded = pad_vector(X_0_unpadded, max_action_dim)  # Pad回32维

        # 对比差异
        diff = (X_0_full - X_0_repadded).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # 后18维的差异
        diff_padding = diff[:,:,action_dim:]
        max_diff_padding = diff_padding.max().item()
        mean_diff_padding = diff_padding.mean().item()

    print(f"\n" + "="*80)
    print("  对比X_0_full vs X_0_repadded")
    print("="*80)

    print(f"\n全部32维:")
    print(f"  最大差异: {max_diff:.6e}")
    print(f"  平均差异: {mean_diff:.6e}")

    print(f"\n后{max_action_dim-action_dim}维 (padding区域):")
    print(f"  最大差异: {max_diff_padding:.6e}")
    print(f"  平均差异: {mean_diff_padding:.6e}")

    print(f"\n" + "="*80)
    print("  结论")
    print("="*80)

    if padding_dims_mean < 1e-6:
        print("✓ Teacher生成的X_0的padding维度全为0")
        print("  → Unpad/pad不会丢失信息")
        print("  → 这不是问题所在")
    else:
        print(f"✗ Teacher生成的X_0的padding维度不为0！")
        print(f"  → 平均绝对值: {padding_dims_mean:.6f}")
        print(f"  → 最大绝对值: {padding_dims_max:.6f}")
        print(f"\n这就是问题所在：")
        print(f"  1. Teacher从X_1_padded(后18维=0)积分时，后18维可能产生非零值")
        print(f"  2. 训练脚本unpad再pad，强制后18维=0")
        print(f"  3. 训练的target(后18维=0) ≠ Teacher实际生成的(后18维≠0)")
        print(f"  4. Student学到了错误的映射！")
        print(f"\n修复方案：")
        print(f"  不要unpad X_0，直接用teacher生成的完整32维X_0训练")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    test_unpad_pad_information_loss(args.teacher_path, args.device)
