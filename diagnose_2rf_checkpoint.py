#!/usr/bin/env python3
"""
Diagnostic script to check 2-RF checkpoint configuration and identify slow inference issues.

Usage:
    python diagnose_2rf_checkpoint.py /path/to/checkpoint/pretrained_model
"""

import sys
import json
from pathlib import Path

def check_checkpoint_config(checkpoint_path):
    """Check configuration in 2-RF checkpoint for common issues."""
    config_path = Path(checkpoint_path) / "config.json"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n" + "="*60)
    print("🔍 2-RF Checkpoint Diagnostic Report")
    print("="*60)
    print(f"\n📁 Checkpoint: {checkpoint_path}")

    # Check critical settings that affect inference speed
    print("\n📊 Configuration Analysis:")
    print("-" * 60)

    issues = []
    warnings = []

    # Check 1: use_reflow
    use_reflow = config.get('use_reflow', False)
    status = "❌" if use_reflow else "✅"
    print(f"{status} use_reflow: {use_reflow}")
    if use_reflow:
        issues.append("use_reflow=True should be False for inference")

    # Check 2: teacher_model_path
    teacher_path = config.get('teacher_model_path')
    status = "❌" if teacher_path else "✅"
    print(f"{status} teacher_model_path: {teacher_path}")
    if teacher_path:
        issues.append("teacher_model_path should be None/null for inference")

    # Check 3: use_cache (CRITICAL for speed!)
    use_cache = config.get('use_cache', True)
    status = "❌" if not use_cache else "✅"
    print(f"{status} use_cache: {use_cache}")
    if not use_cache:
        issues.append("⚠️  CRITICAL: use_cache=False causes 50x slowdown!")

    # Check 4: num_steps
    num_steps = config.get('num_steps', 10)
    status = "ℹ️ "
    print(f"{status} num_steps: {num_steps}")
    if num_steps > 5:
        warnings.append(f"num_steps={num_steps} is high for 2-RF (typically 2-5)")

    # Additional info
    print(f"\nℹ️  chunk_size: {config.get('chunk_size', 'N/A')}")
    print(f"ℹ️  device: {config.get('device', 'N/A')}")

    # Summary
    print("\n" + "="*60)
    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No critical issues found!")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")

    # Recommendations
    print("\n" + "="*60)
    print("💡 Recommendations:")
    print("-" * 60)

    if issues or warnings:
        print("\n🔧 To fix these issues:")
        print("   1. Retrain 2-RF with the updated _save_pretrained method")
        print("      (Already committed to your branch)")
        print("   2. Or manually edit config.json:")
        print('      - Set "use_reflow": false')
        print('      - Set "teacher_model_path": null')
        if not use_cache:
            print('      - Set "use_cache": true  ⚠️  CRITICAL FOR SPEED!')
        print("\n   The updated code will save clean checkpoints automatically.")
    else:
        print("   ✅ Configuration is good!")
        print("   If inference is still slow, check:")
        print("   - GPU utilization")
        print("   - Batch size")
        print("   - Network latency (if loading from remote)")

    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python diagnose_2rf_checkpoint.py <checkpoint_path>")
        print("\nExample:")
        print("  python diagnose_2rf_checkpoint.py /pfs/pfs-ilWc5D/ziqianwang/2rf_put_bottles_dustbin/checkpoints/last/pretrained_model")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    check_checkpoint_config(checkpoint_path)
