#!/usr/bin/env python3
"""
Detailed profiling script to find the cause of slow inference in 2-RF checkpoint.

This script will:
1. Load the checkpoint
2. Profile each step of inference
3. Check if teacher model is being loaded during inference
4. Measure actual inference time

Usage:
    python profile_2rf_inference.py /path/to/checkpoint/pretrained_model
"""

import sys
import os
import time
import torch
import json
from pathlib import Path
import traceback

def profile_checkpoint_loading(checkpoint_path):
    """Profile the checkpoint loading process."""
    print("\n" + "="*70)
    print("📊 PROFILING CHECKPOINT LOADING")
    print("="*70)

    # Check config
    config_path = Path(checkpoint_path) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\n📋 Config:")
    print(f"  - use_reflow: {config.get('use_reflow')}")
    print(f"  - teacher_model_path: {config.get('teacher_model_path')}")
    print(f"  - use_cache: {config.get('use_cache')}")
    print(f"  - num_steps: {config.get('num_steps')}")

    # Check checkpoint size
    model_file = Path(checkpoint_path) / "model.safetensors"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"\n📦 Checkpoint size: {size_mb:.2f} MB")
        if size_mb > 2000:
            print(f"  ⚠️  Very large! Contains teacher model weights?")

    # Profile loading
    print("\n⏱️  Loading checkpoint...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    start = time.time()
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    load_time = time.time() - start

    print(f"  ✅ Loaded in {load_time:.2f}s")

    # Check if teacher model was loaded
    has_teacher = hasattr(policy.model, '_teacher_model') and policy.model._teacher_model is not None
    print(f"\n🔍 Teacher model loaded during init: {has_teacher}")
    if has_teacher:
        print(f"  ⚠️  WARNING: Teacher model should NOT be loaded for inference!")
        print(f"  This adds ~1GB memory and may slow things down")

    return policy

def profile_inference(policy, num_runs=10):
    """Profile actual inference speed."""
    print("\n" + "="*70)
    print("⚡ PROFILING INFERENCE SPEED")
    print("="*70)

    device = next(policy.parameters()).device
    print(f"\n🎯 Device: {device}")
    print(f"📊 Config:")
    print(f"  - num_steps: {policy.config.num_steps}")
    print(f"  - use_cache: {policy.config.use_cache}")
    print(f"  - chunk_size: {policy.config.chunk_size}")

    # Create dummy batch
    batch_size = 1
    height, width = 224, 224

    # Dummy observations
    batch = {
        "observation.state": torch.randn(batch_size, 14, device=device),
        "observation.images.head_camera": torch.randn(batch_size, 3, height, width, device=device),
        "observation.images.left_camera": torch.randn(batch_size, 3, height, width, device=device),
        "observation.images.right_camera": torch.randn(batch_size, 3, height, width, device=device),
        "task": "test task",
    }

    # Prepare batch
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(Path(sys.argv[1]).parent),
    )

    batch = preprocessor(batch)

    policy.eval()
    policy.reset()

    # Warmup
    print("\n🔥 Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = policy.predict_action_chunk(batch)

    # Profile each component
    print("\n📏 Profiling components:")

    # 1. Image encoding
    images, img_masks = policy.prepare_images(batch)
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _, _ = policy.prepare_images(batch)
    img_time = (time.time() - start) / num_runs * 1000
    print(f"  - Image prep: {img_time:.2f}ms")

    # 2. Full inference
    times = []
    print(f"\n⏱️  Running {num_runs} inference iterations...")

    for i in range(num_runs):
        policy.reset()
        start = time.time()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n📊 Results:")
    print(f"  - Average: {avg_time:.2f}ms")
    print(f"  - Min: {min_time:.2f}ms")
    print(f"  - Max: {max_time:.2f}ms")

    # Expected time for different configurations
    expected_time = {
        2: 50,   # 2 steps: ~50ms
        5: 100,  # 5 steps: ~100ms
        10: 200, # 10 steps: ~200ms
    }

    num_steps = policy.config.num_steps
    expected = expected_time.get(num_steps, num_steps * 20)

    print(f"\n🎯 Expected time for {num_steps} steps: ~{expected}ms")

    if avg_time > expected * 2:
        slowdown = avg_time / expected
        print(f"\n❌ SLOW! {slowdown:.1f}x slower than expected")
        print(f"\n🔍 Possible causes:")
        print(f"  1. Teacher model loaded in memory (check above)")
        print(f"  2. GPU not utilized properly")
        print(f"  3. Network/IO bottleneck")
        print(f"  4. Something wrong in forward pass")

        # Check GPU utilization
        if device.type == 'cuda':
            print(f"\n💾 GPU Memory:")
            print(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"  - Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    elif avg_time > expected * 1.2:
        print(f"\n⚠️  Slightly slower than expected ({avg_time/expected:.1f}x)")
    else:
        print(f"\n✅ Performance is normal!")

    return avg_time

def check_forward_path(policy):
    """Check which forward method is being used."""
    print("\n" + "="*70)
    print("🔍 CHECKING FORWARD PATH")
    print("="*70)

    # Check if forward_reflow is being called during inference
    original_forward_reflow = policy.model.forward_reflow
    reflow_called = [False]

    def tracked_forward_reflow(*args, **kwargs):
        reflow_called[0] = True
        return original_forward_reflow(*args, **kwargs)

    policy.model.forward_reflow = tracked_forward_reflow

    # Do an inference
    device = next(policy.parameters()).device
    batch = {
        "observation.state": torch.randn(1, 14, device=device),
        "observation.images.head_camera": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.left_camera": torch.randn(1, 3, 224, 224, device=device),
        "observation.images.right_camera": torch.randn(1, 3, 224, 224, device=device),
        "task": "test",
    }

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(Path(sys.argv[1]).parent),
    )
    batch = preprocessor(batch)

    policy.reset()
    with torch.no_grad():
        _ = policy.predict_action_chunk(batch)

    if reflow_called[0]:
        print("\n❌ ERROR: forward_reflow is being called during inference!")
        print("  This should NEVER happen. It means:")
        print("  1. Something is wrong with the code path")
        print("  2. Training forward is being used instead of inference")
    else:
        print("\n✅ Correct: forward_reflow is NOT called during inference")
        print("  Inference uses sample_actions → denoise_step (correct path)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python profile_2rf_inference.py <checkpoint_path>")
        print("\nExample:")
        print("  python profile_2rf_inference.py /pfs/pfs-ilWc5D/ziqianwang/2rf_put_bottles_dustbin/checkpoints/last/pretrained_model")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    try:
        # 1. Profile loading
        policy = profile_checkpoint_loading(checkpoint_path)

        # 2. Profile inference
        avg_time = profile_inference(policy, num_runs=10)

        # 3. Check forward path
        check_forward_path(policy)

        print("\n" + "="*70)
        print("📋 SUMMARY")
        print("="*70)
        print(f"  Average inference time: {avg_time:.2f}ms")
        print(f"  If this is >1000ms, there's a serious problem!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        traceback.print_exc()
