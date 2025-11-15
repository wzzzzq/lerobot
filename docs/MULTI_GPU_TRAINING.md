# Multi-GPU Training Guide

LeRobot supports multi-GPU training through [HuggingFace Accelerate](https://huggingface.co/docs/accelerate), which provides a flexible and user-friendly interface for distributed training.

## Quick Start

### Using Accelerate (Recommended)

Accelerate is already integrated into `lerobot_train.py` - no code changes needed!

#### Setup Accelerate (One-time)

```bash
# Install accelerate if not already installed
pip install accelerate

# Configure accelerate with default settings
accelerate config default

# Or configure interactively for custom settings
accelerate config
```

#### Launch Training

```bash
# Auto-detect all GPUs and launch
accelerate launch src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset

# Specify number of GPUs
accelerate launch --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset

# With mixed precision (recommended for faster training)
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset
```

### Using Example Scripts

For convenience, we provide ready-to-use example scripts:

```bash
# For SmolVLA training
bash examples/train_smolvla_multi_gpu.sh

# Or use the general template
cp examples/train_multi_gpu_template.sh my_training.sh
# Edit my_training.sh with your configuration
bash my_training.sh
```

See [examples/MULTI_GPU_EXAMPLES.md](../examples/MULTI_GPU_EXAMPLES.md) for detailed usage.

## Configuration Tips

### Batch Size

When using multi-GPU training, the **effective batch size** = `batch_size` × `num_gpus`

```yaml
# Example: 4 GPUs with batch_size=32
# Effective batch size = 32 × 4 = 128

batch_size: 32  # Per-GPU batch size
```

**Important**: When migrating from single-GPU to multi-GPU, adjust batch size to maintain the same effective batch size:

```bash
# Single GPU: batch_size=64
# 4 GPUs: batch_size=16 (effective = 16 × 4 = 64)
```

### Learning Rate Scaling

When changing the effective batch size, you typically need to scale the learning rate:

```yaml
# Single GPU: batch_size=64, lr=1e-4
# 4 GPUs: batch_size=16 per GPU (effective 64), lr=1e-4 (same)
# 4 GPUs: batch_size=32 per GPU (effective 128), lr=2e-4 (scaled)
```

**Rule of thumb**: If you double the effective batch size, increase learning rate by √2 to 2×.

### Number of Workers

Adjust `num_workers` based on available CPU cores:

```yaml
# For 4 GPUs with 32 CPU cores
num_workers: 8  # 8 workers per GPU = 32 total workers

# Recommended range: 4-8 workers per GPU
```

### Specifying GPUs

```bash
# Use specific GPUs (e.g., GPU 2 and 3)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes=2 \
    src/lerobot/scripts/lerobot_train.py [args]

# Use GPUs 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py [args]
```

## Performance Optimization

### 1. Enable Mixed Precision Training

Mixed precision (FP16/BF16) can provide 1.5-2× speedup:

```bash
# BF16 (recommended for Ampere GPUs: A100, RTX 3090, RTX 4090)
accelerate launch --mixed_precision=bf16 \
    src/lerobot/scripts/lerobot_train.py [args]

# FP16 (for older GPUs: V100, RTX 2080 Ti)
accelerate launch --mixed_precision=fp16 \
    src/lerobot/scripts/lerobot_train.py [args]
```

### 2. Optimize Batch Size

Find the maximum batch size that fits in GPU memory:

```bash
# Start small and increase gradually: 8 → 16 → 24 → 32
batch_size: 24  # Adjust based on GPU memory
```

### 3. Adjust Data Loading Workers

```bash
# CPU cores sufficient: use more workers
num_workers: 8  # Per GPU

# CPU cores limited: use fewer workers
num_workers: 4  # Per GPU
```

### 4. Enable Efficient Backends

These optimizations are already included in `lerobot_train.py`:

```python
torch.backends.cudnn.benchmark = True  # Auto-select best convolution algorithm
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
```

## Monitoring Multi-GPU Training

### Check GPU Utilization

```bash
# Real-time monitoring with nvidia-smi
watch -n 1 nvidia-smi

# Or use gpustat (more user-friendly)
pip install gpustat
gpustat -i 1

# Monitor specific GPUs
nvidia-smi dmon -i 0,1,2,3
```

**Expected GPU utilization**:
- ✅ **Good**: 80-100% utilization on all GPUs
- ⚠️ **Okay**: 50-80% (consider increasing batch size)
- ❌ **Poor**: <50% (data loading bottleneck or batch size too small)

### WandB Logging

Only the main process (rank 0) logs to WandB to avoid duplicate entries:

```bash
accelerate launch \
    src/lerobot/scripts/lerobot_train.py \
    wandb.enable=true \
    wandb.project=robot-learning \
    wandb.entity=your-username
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:

```yaml
# 1. Reduce per-GPU batch size
batch_size: 8  # Reduce from 16

# 2. Enable mixed precision
# Use: --mixed_precision=bf16

# 3. Reduce number of workers
num_workers: 4  # Reduce from 8
```

### Issue: NCCL Errors

**Solution**: Set environment variables to disable problematic features:

```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG=INFO  # For debugging
```

Add these to your training script or shell configuration.

### Issue: Low GPU Utilization

**Diagnosis**:

```bash
# Check GPU utilization
nvidia-smi

# If utilization < 80%, try:
# 1. Increase batch_size
# 2. Increase num_workers
# 3. Check if data loading is the bottleneck
```

### Issue: Uneven GPU Utilization

**Cause**: Usually hardware differences or data distribution issues

**Solutions**:
1. Ensure all GPUs are the same model
2. Verify `DistributedSampler` is being used (automatic in lerobot_train.py)
3. Check for data preprocessing bottlenecks

### Issue: Slower than Single GPU

**Possible causes**:
1. Communication overhead (model/batch size too small)
2. Slow GPU interconnect
3. Data loading bottleneck

**Solutions**:
1. Increase batch size per GPU
2. Increase num_workers
3. Use mixed precision training
4. Check GPU interconnect (use `nvidia-smi topo -m`)

## Advanced: Multi-Node Training

For training across multiple machines with Accelerate:

```bash
# On master node (machine rank 0):
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=0 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    src/lerobot/scripts/lerobot_train.py [args]

# On worker node (machine rank 1):
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    src/lerobot/scripts/lerobot_train.py [args]
```

## Example: Training SmolVLA on Multiple GPUs

### Basic 4-GPU Training

```bash
accelerate launch \
    --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=16 \
    output_dir=outputs/smolvla_4gpu
```

### Optimized 4-GPU Training with Mixed Precision

```bash
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=16 \
    num_workers=8 \
    wandb.enable=true \
    wandb.project=robot-learning \
    output_dir=outputs/smolvla_4gpu_bf16
```

### Using Specific GPUs

```bash
# Train on GPU 2,3,4,5 only
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/lerobot/scripts/lerobot_train.py \
    policy.path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=16
```

## Performance Benchmarks

Example training speedup on NVIDIA A100 GPUs (SmolVLA model):

| GPUs | Batch Size per GPU | Effective Batch Size | Speedup | Efficiency |
|------|-------------------|---------------------|---------|------------|
| 1    | 32                | 32                  | 1.0×    | 100%       |
| 2    | 32                | 64                  | 1.9×    | 95%        |
| 4    | 32                | 128                 | 3.7×    | 93%        |
| 8    | 32                | 256                 | 5.4×    | 68%        |

**Note**:
- Efficiency = Speedup / Number of GPUs
- Efficiency decreases with more GPUs due to communication overhead
- Actual performance depends on model size, batch size, and hardware

## Best Practices

1. **Start with small-scale testing**
   ```bash
   # Test configuration with few steps first
   steps: 100
   # Then run full training after confirming no issues
   ```

2. **Keep effective batch size consistent**
   ```bash
   # When scaling from 1 to 4 GPUs:
   # Single GPU: batch_size=64
   # 4 GPUs: batch_size=16 (effective = 64)
   ```

3. **Monitor all GPUs**
   ```bash
   # Ensure all GPUs have similar utilization
   watch -n 1 nvidia-smi
   ```

4. **Use WandB for experiment tracking**
   ```bash
   wandb.enable=true
   wandb.run_id="run_$(date +%Y%m%d_%H%M%S)"
   ```

5. **Save checkpoints regularly**
   ```bash
   save_freq=1000  # Save every 1000 steps
   ```

## How It Works

`lerobot_train.py` automatically handles multi-GPU training through Accelerate:

✅ Auto-detects multiple GPUs
✅ Distributes batches across GPUs
✅ Synchronizes gradients
✅ Handles mixed precision
✅ Uses DistributedSampler for data loading
✅ Saves checkpoints only on main process
✅ Logs to WandB only on main process
✅ Aggregates metrics across processes

**No code changes needed - just use Accelerate to launch!**

## References

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Accelerate Quick Tour](https://huggingface.co/docs/accelerate/quicktour)
- [Launching Distributed Training](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
