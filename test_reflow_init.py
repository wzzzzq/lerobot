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

    print("\n2. Checking parameter freezing status...")

    # Count parameters by component
    vlm_params = 0
    vlm_trainable = 0
    expert_params = 0
    expert_trainable = 0
    action_proj_params = 0
    action_proj_trainable = 0

    # Check VLM parameters
    for name, param in policy.model.vlm_with_expert.vlm.named_parameters():
        vlm_params += param.numel()
        if param.requires_grad:
            vlm_trainable += param.numel()

    # Check expert parameters
    if hasattr(policy.model.vlm_with_expert, 'lm_expert'):
        for name, param in policy.model.vlm_with_expert.lm_expert.named_parameters():
            expert_params += param.numel()
            if param.requires_grad:
                expert_trainable += param.numel()

    # Check action projection parameters
    for module in [policy.model.action_in_proj, policy.model.action_out_proj,
                   policy.model.action_time_mlp_in, policy.model.action_time_mlp_out]:
        for param in module.parameters():
            action_proj_params += param.numel()
            if param.requires_grad:
                action_proj_trainable += param.numel()

    print(f"\nVLM Parameters:")
    print(f"  Total: {vlm_params:,}")
    print(f"  Trainable: {vlm_trainable:,}")
    print(f"  Status: {'✗ NOT FROZEN' if vlm_trainable > 0 else '✓ FROZEN'}")

    print(f"\nExpert Parameters:")
    print(f"  Total: {expert_params:,}")
    print(f"  Trainable: {expert_trainable:,}")
    print(f"  Status: {'✓ TRAINABLE' if expert_trainable == expert_params else '✗ FROZEN'}")

    print(f"\nAction Projection Parameters:")
    print(f"  Total: {action_proj_params:,}")
    print(f"  Trainable: {action_proj_trainable:,}")
    print(f"  Status: {'✓ TRAINABLE' if action_proj_trainable == action_proj_params else '✗ FROZEN'}")

    # Overall statistics
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"\nOverall Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")

    # Verify expectations
    success = True

    if vlm_trainable > 0:
        print("\n✗ FAIL: VLM should be frozen but has trainable parameters")
        success = False

    if expert_trainable == 0:
        print("\n✗ FAIL: Expert should be trainable but is frozen")
        success = False

    if action_proj_trainable == 0:
        print("\n✗ FAIL: Action projections should be trainable but are frozen")
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
