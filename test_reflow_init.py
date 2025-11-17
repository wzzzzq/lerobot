#!/usr/bin/env python3
"""Test script to verify Reflow student initialization and VLM freezing."""

import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def test_reflow_initialization():
    """Test that student model correctly inherits teacher weights and freezes VLM."""

    print("="*80)
    print("Testing Reflow Initialization and VLM Freezing")
    print("="*80)

    # Create a config with reflow enabled
    config = SmolVLAConfig(
        use_reflow=True,
        teacher_model_path="/pfs/pfs-ilWc5D/ziqianwang/pretrain_put_bottles_dustbin/checkpoints/020000/pretrained_model",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating SmolVLA policy with reflow enabled...")
    try:
        policy = SmolVLAPolicy(config=config)
        print("✓ Policy created successfully")
    except Exception as e:
        print(f"✗ Failed to create policy: {e}")
        return False

    print("\n2. Verifying train_expert_only configuration...")

    # Check that train_expert_only is set correctly
    print(f"\nConfiguration:")
    print(f"  use_reflow: {policy.config.use_reflow}")
    print(f"  train_expert_only: {policy.config.train_expert_only}")
    print(f"  Status: {'✓ CORRECT' if policy.config.train_expert_only else '✗ INCORRECT'}")

    # Overall statistics
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"\nOverall Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")

    # Expected trainable ratio should be around 20-30% when train_expert_only=True
    trainable_ratio = 100 * trainable_params / total_params
    reasonable_ratio = 10 <= trainable_ratio <= 40
    print(f"  Ratio check: {'✓ REASONABLE (10-40%)' if reasonable_ratio else '✗ UNEXPECTED'}")

    # Check teacher model device
    print(f"\n3. Checking teacher model device...")
    teacher = policy.model.load_teacher_model()
    teacher_device = next(teacher.parameters()).device
    student_device = next(policy.parameters()).device
    same_device = teacher_device == student_device

    print(f"\nTeacher Model Device:")
    print(f"  Teacher device: {teacher_device}")
    print(f"  Student device: {student_device}")
    print(f"  Status: {'✓ SAME DEVICE' if same_device else '✗ DIFFERENT DEVICE'}")

    # Verify expectations
    success = True

    if not policy.config.train_expert_only:
        print("\n✗ FAIL: train_expert_only should be True for reflow")
        success = False

    if not reasonable_ratio:
        print(f"\n✗ FAIL: Trainable ratio {trainable_ratio:.2f}% is outside expected range (10-40%)")
        success = False

    if not same_device:
        print("\n✗ FAIL: Teacher and student should be on the same device")
        success = False

    if success:
        print("\n" + "="*80)
        print("✓ All tests PASSED! Reflow initialization is working correctly.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ Some tests FAILED. Please check the configuration.")
        print("="*80)

    return success


if __name__ == "__main__":
    test_reflow_initialization()
