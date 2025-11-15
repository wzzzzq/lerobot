# Multi-GPU Training Quick Start

## 🚀 Fastest Way to Start

```bash
# 1. Make the script executable (one-time setup)
chmod +x scripts/train_multi_gpu.sh

# 2. Run training on all available GPUs
./scripts/train_multi_gpu.sh --config your_config.yaml

# 3. Or specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_multi_gpu.sh --config your_config.yaml
```

That's it! The script will automatically:
- Detect available GPUs
- Set up distributed training
- Launch training with optimal settings

## 📁 Files Added

### 1. Training Scripts
- **`src/lerobot/scripts/lerobot_train_ddp.py`** - Native PyTorch DDP training script
  - Alternative to Accelerate-based training
  - More control over distributed settings
  - Use with `torchrun` command

### 2. Launch Scripts
- **`scripts/train_multi_gpu.sh`** - Convenience wrapper for multi-GPU training
  - Auto-detects GPUs
  - Handles both Accelerate and native DDP
  - Simplifies command-line usage

### 3. Documentation
- **`docs/MULTI_GPU_TRAINING.md`** - Comprehensive multi-GPU training guide
  - Setup instructions
  - Performance optimization tips
  - Troubleshooting guide
  - Multi-node training examples

## 📊 Quick Comparison

| Method | Command | Pros | Cons |
|--------|---------|------|------|
| **Convenience Script** | `./scripts/train_multi_gpu.sh` | Easiest, auto-detects GPUs | Less control |
| **Accelerate** | `accelerate launch lerobot_train.py` | Flexible, easy multi-node | Requires setup |
| **Native DDP** | `torchrun lerobot_train_ddp.py` | Full control, no dependencies | More complex |

## 💡 Tips

### Adjust Batch Size
```yaml
# Single GPU: batch_size=64
# 4 GPUs: batch_size=16 (effective batch_size = 16 × 4 = 64)
batch_size: 16
```

### Enable Mixed Precision (Faster Training)
```bash
# With Accelerate
accelerate launch --mixed_precision=bf16 src/lerobot/scripts/lerobot_train.py ...

# With convenience script (edit train_multi_gpu.sh to enable)
# Line 62: --mixed_precision=bf16 \
```

### Monitor GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or install gpustat
pip install gpustat
gpustat -i 1
```

## 🔧 Troubleshooting

### Out of Memory?
1. Reduce `batch_size` in config
2. Reduce `num_workers`
3. Enable mixed precision

### Slow Training?
1. Increase `batch_size` per GPU
2. Increase `num_workers` (e.g., 4-8 per GPU)
3. Check GPU utilization with `nvidia-smi`

### GPUs Not Detected?
```bash
# Check visible GPUs
nvidia-smi

# Explicitly set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 📖 Full Documentation

For detailed information, see [`docs/MULTI_GPU_TRAINING.md`](docs/MULTI_GPU_TRAINING.md)

## 🎯 Example: Training SmolVLA with 4 GPUs

```bash
# Method 1: Using convenience script (easiest)
./scripts/train_multi_gpu.sh \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=christianwang-sjtu/so100-red-dustbin \
    batch_size=12 \
    output_dir=outputs/smolvla_4gpu

# Method 2: Using Accelerate directly
accelerate launch --num_processes=4 --mixed_precision=bf16 \
    src/lerobot/scripts/lerobot_train.py \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=christianwang-sjtu/so100-red-dustbin \
    batch_size=12

# Method 3: Using native DDP
torchrun --nproc_per_node=4 \
    src/lerobot/scripts/lerobot_train_ddp.py \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=christianwang-sjtu/so100-red-dustbin \
    batch_size=12
```

## ⚠️ Important Notes

1. **Effective Batch Size**: When using N GPUs with batch_size=B, effective batch size = B × N
2. **Learning Rate**: May need adjustment when changing effective batch size
3. **Random Seed**: Each GPU gets a different seed (seed + rank) for proper shuffling
4. **Checkpointing**: Only rank 0 saves checkpoints to avoid conflicts
5. **WandB Logging**: Only rank 0 logs to avoid duplicate entries

## 🎓 Next Steps

- Read the full guide: `docs/MULTI_GPU_TRAINING.md`
- Experiment with different batch sizes and learning rates
- Try mixed precision training for faster convergence
- Monitor training with WandB or TensorBoard
