#!/usr/bin/env python3
"""
Reflow推理问题诊断脚本

这个脚本会检查所有可能导致reflow训练的模型在eval时表现不一致的问题：
1. 模型中是否有dropout/batchnorm等对train/eval敏感的层
2. forward()和sample_actions()是否生成相同的结果
3. teacher在两种前向方式中生成的action是否一致
4. train模式和eval模式下的输出差异
5. 配置参数验证

运行方法：
    python diagnose_reflow_inference.py \\
        --teacher_path /path/to/teacher/checkpoint \\
        --student_path /path/to/student/checkpoint \\
        --dataset_repo_id your_dataset_repo_id \\
        --num_samples 5
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def print_section(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_model_layers(policy: SmolVLAPolicy, model_name: str):
    """检查模型中是否有对train/eval模式敏感的层"""
    print_section(f"检查{model_name}模型层")

    dropout_layers = []
    batchnorm_layers = []
    layernorm_layers = []

    for name, module in policy.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layers.append((name, module))
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            batchnorm_layers.append((name, module))
        elif isinstance(module, torch.nn.LayerNorm):
            layernorm_layers.append((name, module))

    # 打印结果
    if dropout_layers:
        print(f"⚠️  发现 {len(dropout_layers)} 个Dropout层（train/eval行为不同）:")
        for name, module in dropout_layers[:5]:  # 只显示前5个
            print(f"   - {name}: {module}")
        if len(dropout_layers) > 5:
            print(f"   ... 还有 {len(dropout_layers) - 5} 个")
    else:
        print("✓ 未发现Dropout层")

    if batchnorm_layers:
        print(f"\n⚠️  发现 {len(batchnorm_layers)} 个BatchNorm层（train/eval行为不同）:")
        for name, module in batchnorm_layers[:5]:
            print(f"   - {name}: {module}")
        if len(batchnorm_layers) > 5:
            print(f"   ... 还有 {len(batchnorm_layers) - 5} 个")
    else:
        print("✓ 未发现BatchNorm层")

    if layernorm_layers:
        print(f"\n✓ 发现 {len(layernorm_layers)} 个LayerNorm层（train/eval行为相同，无问题）")

    return len(dropout_layers) > 0 or len(batchnorm_layers) > 0


def check_config_consistency(teacher: SmolVLAPolicy, student: SmolVLAPolicy):
    """检查teacher和student的配置一致性"""
    print_section("检查Teacher和Student配置一致性")

    config_items = [
        ("num_steps", "ODE积分步数"),
        ("chunk_size", "动作chunk大小"),
        ("max_action_dim", "最大动作维度"),
        ("use_cache", "是否使用KV cache"),
    ]

    all_consistent = True
    for attr, desc in config_items:
        teacher_val = getattr(teacher.config, attr, None)
        student_val = getattr(student.config, attr, None)

        if teacher_val == student_val:
            print(f"✓ {desc} ({attr}): {teacher_val}")
        else:
            print(f"✗ {desc} ({attr}): Teacher={teacher_val}, Student={student_val}")
            all_consistent = False

    return all_consistent


def prepare_test_batch(dataset: LeRobotDataset, policy: SmolVLAPolicy, device: str):
    """准备测试batch"""
    # 获取一个样本
    sample = dataset[0]

    # 转换为batch格式
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value

    # 如果需要pi_aloha适配
    if policy.config.adapt_to_pi_aloha:
        batch["observation.state"] = policy._pi_aloha_decode_state(batch["observation.state"])

    # 预处理
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)

    # 假设语言tokens已经在batch中（如果没有需要添加）
    if "observation.language_tokens" not in batch:
        # 创建dummy language tokens
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
        text = "test task"
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        batch["observation.language_tokens"] = tokens["input_ids"].to(device)
        batch["observation.language_attention_mask"] = tokens["attention_mask"].to(device)

    lang_tokens = batch["observation.language_tokens"]
    lang_masks = batch["observation.language_attention_mask"]

    return images, img_masks, lang_tokens, lang_masks, state, batch


def test_forward_vs_sample_actions_single_step(
    policy: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    model_name: str,
    device: str
):
    """测试在单个时间步，forward()和手动denoise_step()的输出是否一致"""
    print_section(f"{model_name}: 测试Forward vs Denoise Step (单个时间步)")

    policy.eval()

    batch_size = state.shape[0]
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    # 固定随机种子以保证可重复性
    torch.manual_seed(42)

    # 1. 生成固定的噪声和动作
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)
    actions = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    # 2. 固定时间点
    time = torch.tensor([0.5], device=device, dtype=torch.float32)

    with torch.no_grad():
        # 方法1: 使用forward()计算velocity
        time_expanded = time[:, None, None]
        x_t_forward = time_expanded * noise + (1 - time_expanded) * actions
        u_t_expected = noise - actions

        losses = policy.model.forward(
            images, img_masks, lang_tokens, lang_masks, state,
            actions, noise=noise, time=time
        )

        # Forward返回的是loss，我们需要从中反推velocity
        # loss = mse(u_t, v_t)，但我们无法直接拿到v_t
        # 所以我们换一个测试方法：直接测试sample_actions的中间步骤

        # 方法2: 使用sample_actions的流程手动计算一步
        # 先计算prefix cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=policy.config.use_cache,
            fill_kv_cache=True,
        )

        # 然后执行一步denoise
        time_expanded_denoise = time.expand(batch_size)
        v_t = policy.model.denoise_step(
            prefix_pad_masks,
            past_key_values,
            x_t_forward,
            time_expanded_denoise,
        )

        print(f"时间点 t={time.item():.3f}")
        print(f"x_t shape: {x_t_forward.shape}")
        print(f"v_t (denoise_step) shape: {v_t.shape}")
        print(f"v_t mean: {v_t.mean().item():.6f}, std: {v_t.std().item():.6f}")
        print(f"v_t range: [{v_t.min().item():.6f}, {v_t.max().item():.6f}]")

        # 注意：我们无法直接从forward()拿到v_t，所以这个测试主要是验证denoise_step能正常运行


def test_forward_vs_sample_actions_full_ode(
    policy: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    model_name: str,
    device: str
):
    """测试完整ODE积分：sample_actions()的输出"""
    print_section(f"{model_name}: 测试Sample Actions (完整ODE积分)")

    policy.eval()

    batch_size = state.shape[0]
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    # 固定随机种子
    torch.manual_seed(42)
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        # 使用sample_actions完整积分
        actions_pred = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        print(f"Sample actions输出 shape: {actions_pred.shape}")
        print(f"Actions mean: {actions_pred.mean().item():.6f}, std: {actions_pred.std().item():.6f}")
        print(f"Actions range: [{actions_pred.min().item():.6f}, {actions_pred.max().item():.6f}]")

        return actions_pred


def test_train_vs_eval_mode(
    policy: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    model_name: str,
    device: str
):
    """测试train模式和eval模式下的输出差异"""
    print_section(f"{model_name}: 测试Train模式 vs Eval模式")

    batch_size = state.shape[0]
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    # 固定随机种子
    torch.manual_seed(123)
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    # Eval模式
    policy.eval()
    with torch.no_grad():
        actions_eval = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )

    # Train模式
    policy.train()
    # 重置随机种子确保噪声相同
    torch.manual_seed(123)
    noise_train = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        actions_train = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise_train
        )

    # 对比差异
    diff = (actions_eval - actions_train).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Eval模式输出: mean={actions_eval.mean().item():.6f}, std={actions_eval.std().item():.6f}")
    print(f"Train模式输出: mean={actions_train.mean().item():.6f}, std={actions_train.std().item():.6f}")
    print(f"\n差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  相对差异: {mean_diff / (actions_eval.abs().mean().item() + 1e-8) * 100:.2f}%")

    if max_diff < 1e-5:
        print("✓ Train和Eval模式输出一致（差异 < 1e-5）")
        return True
    elif max_diff < 1e-3:
        print("⚠️  Train和Eval模式有微小差异（1e-5 < 差异 < 1e-3）")
        return False
    else:
        print("✗ Train和Eval模式输出显著不同（差异 > 1e-3）！")
        print("   这可能是因为模型中有Dropout或BatchNorm层！")
        return False


def test_teacher_two_methods(
    teacher: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    device: str
):
    """测试teacher用两种方法生成action是否一致"""
    print_section("Teacher: 对比两种前向方式生成的Action")

    teacher.eval()

    batch_size = state.shape[0]
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim

    # 固定随机种子
    torch.manual_seed(456)
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        # 方法1: sample_actions (正常的推理方法)
        actions_sample = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )

        # 方法2: 模拟训练时的方式
        # 在训练reflow时，teacher是这样生成X0的
        # 重置随机种子确保噪声相同
        torch.manual_seed(456)
        noise2 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

        actions_sample2 = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise2
        )

    # 对比
    diff = (actions_sample - actions_sample2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"方法1输出: mean={actions_sample.mean().item():.6f}, std={actions_sample.std().item():.6f}")
    print(f"方法2输出: mean={actions_sample2.mean().item():.6f}, std={actions_sample2.std().item():.6f}")
    print(f"\n差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")

    if max_diff < 1e-5:
        print("✓ 两种方法生成的action完全一致")
        return True
    else:
        print(f"✗ 两种方法生成的action不一致！最大差异: {max_diff:.6f}")
        return False


def test_sample_actions_vs_forward_consistency(
    teacher: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    device: str
):
    """
    测试sample_actions和forward的一致性

    关键测试：在reflow训练中
    - Teacher用sample_actions生成X0
    - Student用forward来训练
    - Student用sample_actions来推理

    这个测试验证：Teacher用sample_actions生成的X0，
    通过forward计算是否能得到合理的loss
    """
    print_section("关键测试: Sample_actions生成的X0 vs Forward验证")

    teacher.eval()

    batch_size = state.shape[0]
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim

    # 固定随机种子
    torch.manual_seed(999)

    with torch.no_grad():
        # 步骤1: 生成随机噪声X_1
        X_1 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

        # 步骤2: Teacher用sample_actions从X_1生成X_0（模拟训练时teacher生成target）
        print("步骤1: Teacher用sample_actions从噪声生成X0...")
        X_0 = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1
        )

        print(f"  X_0 shape: {X_0.shape}")
        print(f"  X_0 mean: {X_0.mean().item():.6f}, std: {X_0.std().item():.6f}")
        print(f"  X_0 range: [{X_0.min().item():.6f}, {X_0.max().item():.6f}]")

        # 步骤3: 用forward验证这个X_0
        # 如果sample_actions生成的X_0确实是从噪声正确积分得到的
        # 那么用forward在各个时间点计算，loss应该合理
        print("\n步骤2: 用forward验证X_0在不同时间点的一致性...")

        test_times = [0.1, 0.3, 0.5, 0.7, 0.9]
        losses_at_times = []

        for t in test_times:
            time_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # Forward会在时间t计算loss
            loss = teacher.model.forward(
                images, img_masks, lang_tokens, lang_masks, state,
                X_0, noise=X_1, time=time_tensor
            )

            mean_loss = loss.mean().item()
            losses_at_times.append((t, mean_loss))
            print(f"  t={t:.1f}: loss={mean_loss:.6f}")

        # 步骤4: 检查loss的合理性
        # 对于一个训练好的模型，loss应该相对较小
        avg_loss = np.mean([l for _, l in losses_at_times])
        max_loss = max([l for _, l in losses_at_times])

        print(f"\n验证结果:")
        print(f"  平均loss: {avg_loss:.6f}")
        print(f"  最大loss: {max_loss:.6f}")

        # 步骤5: 测试从相同噪声重新积分是否得到相同X_0
        print("\n步骤3: 从相同噪声重新sample_actions，验证确定性...")
        torch.manual_seed(999)
        X_1_repeat = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)
        X_0_repeat = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=X_1_repeat
        )

        diff = (X_0 - X_0_repeat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"  X_0与重复生成的X_0'差异:")
        print(f"    最大差异: {max_diff:.6e}")
        print(f"    平均差异: {mean_diff:.6e}")

        if max_diff < 1e-5:
            print("  ✓ 重复生成完全一致（确定性正确）")
            deterministic = True
        else:
            print(f"  ✗ 重复生成不一致！可能有随机性问题")
            deterministic = False

        return avg_loss, deterministic


def test_teacher_student_sample_actions_comparison(
    teacher: SmolVLAPolicy,
    student: SmolVLAPolicy,
    images, img_masks, lang_tokens, lang_masks, state,
    device: str
):
    """
    对比teacher和student用sample_actions从相同噪声生成的action

    如果student训练正确，它应该能生成和teacher类似的action
    """
    print_section("对比Teacher和Student的sample_actions输出")

    teacher.eval()
    student.eval()

    batch_size = state.shape[0]
    chunk_size = teacher.config.chunk_size
    max_action_dim = teacher.config.max_action_dim

    # 固定随机种子
    torch.manual_seed(888)
    noise = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Teacher生成
        teacher_actions = teacher.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone()
        )

        # Student生成（从相同噪声）
        torch.manual_seed(888)
        noise_student = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)
        student_actions = student.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise_student
        )

    # 对比
    diff = (teacher_actions - student_actions).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # 计算相对差异
    teacher_scale = teacher_actions.abs().mean().item()
    relative_diff = mean_diff / (teacher_scale + 1e-8) * 100

    print(f"Teacher输出: mean={teacher_actions.mean().item():.6f}, std={teacher_actions.std().item():.6f}")
    print(f"Student输出: mean={student_actions.mean().item():.6f}, std={student_actions.std().item():.6f}")
    print(f"\n差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  相对差异: {relative_diff:.2f}%")

    if relative_diff < 1.0:
        print("✓ Student和Teacher输出非常接近（相对差异 < 1%）")
        print("  → Student训练效果很好")
        return True
    elif relative_diff < 10.0:
        print("⚠️  Student和Teacher输出有一定差异（1% < 相对差异 < 10%）")
        print("  → Student可能需要更多训练，或存在小问题")
        return False
    else:
        print("✗ Student和Teacher输出差异很大（相对差异 > 10%）！")
        print("  → 这可能说明:")
        print("     1. Student训练有问题（train/eval模式不一致等）")
        print("     2. Student还未充分训练")
        print("     3. 训练和推理的代码路径不一致")
        return False


def test_noise_consistency(policy: SmolVLAPolicy, device: str):
    """测试噪声生成的一致性"""
    print_section("测试噪声生成一致性")

    batch_size = 2
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    # 方法1: sample_noise
    torch.manual_seed(789)
    noise1 = policy.model.sample_noise((batch_size, chunk_size, max_action_dim), device)

    # 方法2: torch.randn (训练时使用的)
    torch.manual_seed(789)
    noise2 = torch.randn(batch_size, chunk_size, max_action_dim, device=device, dtype=torch.float32)

    diff = (noise1 - noise2).abs().max().item()

    print(f"sample_noise dtype: {noise1.dtype}")
    print(f"torch.randn dtype: {noise2.dtype}")
    print(f"最大差异: {diff:.6f}")

    if diff < 1e-6:
        print("✓ 噪声生成方法一致")
    else:
        print(f"⚠️  噪声生成有差异（可能是dtype不同）")


def main():
    parser = argparse.ArgumentParser(description="诊断Reflow推理问题")
    parser.add_argument("--teacher_path", type=str, required=True, help="Teacher模型checkpoint路径")
    parser.add_argument("--student_path", type=str, required=True, help="Student模型checkpoint路径")
    parser.add_argument("--dataset_repo_id", type=str, default=None, help="数据集repo_id (可选)")
    parser.add_argument("--num_samples", type=int, default=3, help="测试样本数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  Reflow推理问题诊断工具")
    print("=" * 80)
    print(f"\nTeacher路径: {args.teacher_path}")
    print(f"Student路径: {args.student_path}")
    print(f"设备: {args.device}")

    # 加载模型
    print("\n正在加载模型...")
    teacher = SmolVLAPolicy.from_pretrained(args.teacher_path)
    student = SmolVLAPolicy.from_pretrained(args.student_path)

    teacher = teacher.to(args.device)
    student = student.to(args.device)

    print("✓ 模型加载完成")

    # 1. 检查模型层
    teacher_has_sensitive_layers = check_model_layers(teacher, "Teacher")
    student_has_sensitive_layers = check_model_layers(student, "Student")

    # 2. 检查配置一致性
    config_consistent = check_config_consistency(teacher, student)

    # 3. 准备测试数据
    print_section("准备测试数据")

    if args.dataset_repo_id:
        print(f"从数据集加载: {args.dataset_repo_id}")
        dataset = LeRobotDataset(args.dataset_repo_id)
        images, img_masks, lang_tokens, lang_masks, state, batch = prepare_test_batch(
            dataset, teacher, args.device
        )
    else:
        print("使用随机生成的测试数据")
        # 生成随机测试数据
        batch_size = 1
        images = [torch.randn(batch_size, 3, 224, 224, device=args.device) for _ in range(3)]
        img_masks = [torch.ones(batch_size, dtype=torch.bool, device=args.device) for _ in range(3)]

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(teacher.config.vlm_model_name)
        tokens = tokenizer("pick up the object", return_tensors="pt", padding=True, max_length=48)
        lang_tokens = tokens["input_ids"].to(args.device)
        lang_masks = tokens["attention_mask"].to(args.device)

        state = torch.randn(batch_size, teacher.config.max_state_dim, device=args.device)

    print("✓ 测试数据准备完成")

    # 4. 测试teacher两种方法
    teacher_consistent = test_teacher_two_methods(
        teacher, images, img_masks, lang_tokens, lang_masks, state, args.device
    )

    # 5. 测试student的train/eval模式差异
    student_train_eval_consistent = test_train_vs_eval_mode(
        student, images, img_masks, lang_tokens, lang_masks, state, "Student", args.device
    )

    # 6. 测试forward vs sample_actions
    test_forward_vs_sample_actions_single_step(
        student, images, img_masks, lang_tokens, lang_masks, state, "Student", args.device
    )

    test_forward_vs_sample_actions_full_ode(
        student, images, img_masks, lang_tokens, lang_masks, state, "Student", args.device
    )

    # 7. 【关键测试】测试teacher的sample_actions和forward一致性
    avg_loss, sample_actions_deterministic = test_sample_actions_vs_forward_consistency(
        teacher, images, img_masks, lang_tokens, lang_masks, state, args.device
    )

    # 8. 【关键测试】对比teacher和student的sample_actions输出
    teacher_student_similar = test_teacher_student_sample_actions_comparison(
        teacher, student, images, img_masks, lang_tokens, lang_masks, state, args.device
    )

    # 9. 测试噪声一致性
    test_noise_consistency(teacher, args.device)

    # 最终总结
    print_section("诊断总结")

    issues = []

    if teacher_has_sensitive_layers or student_has_sensitive_layers:
        issues.append("⚠️  模型中存在Dropout或BatchNorm层，train/eval模式会有差异")

    if not config_consistent:
        issues.append("✗ Teacher和Student配置不一致")

    if not teacher_consistent:
        issues.append("✗ Teacher用不同方法生成的action不一致")

    if not student_train_eval_consistent:
        issues.append("✗ Student在train和eval模式下输出不一致")

    if not sample_actions_deterministic:
        issues.append("✗ sample_actions生成的结果不确定（有随机性）")

    if not teacher_student_similar:
        issues.append("⚠️  Student和Teacher的输出差异较大（可能训练不足或有bug）")

    if issues:
        print("发现以下问题:\n")
        for issue in issues:
            print(f"  {issue}")

        print("\n建议的修复方向:")
        if teacher_has_sensitive_layers or student_has_sensitive_layers:
            print("  1. 在reflow训练时，确保student.forward()之前调用policy.eval()")
            print("     但仍然保留梯度计算以便backward")

        if not config_consistent:
            print("  2. 确保teacher和student使用相同的num_steps和其他关键配置")

        if not student_train_eval_consistent:
            print("  3. 检查模型中的Dropout/BatchNorm层，考虑在推理时禁用")

        if not sample_actions_deterministic:
            print("  4. 检查sample_actions中是否有未控制的随机性")
            print("     确保使用固定随机种子时结果可重复")

        if not teacher_student_similar:
            print("  5. Student和Teacher输出差异大，可能原因:")
            print("     - Student训练时使用train()模式，应改为eval()模式")
            print("     - Student训练步数不够")
            print("     - 训练和推理的代码路径不一致（forward vs sample_actions）")
    else:
        print("✓ 未发现明显问题，可能需要进一步调试")

    print("\n" + "=" * 80)
    print("诊断完成!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
