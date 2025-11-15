# Multi-GPU Training Guide

LeRobot supports multi-GPU training through two approaches:

1. **Accelerate-based training** (Recommended) - `lerobot_train.py`
2. **Native PyTorch DDP** - `lerobot_train_ddp.py`

## Quick Start

### Method 1: Using the Convenience Script (Easiest)

```bash
# Make the script executable
chmod +x scripts/train_multi_gpu.sh

# Train with all available GPUs
./scripts/train_multi_gpu.sh --config configs/your_config.yaml

# Train with specific GPUs (e.g., GPU 0 and 1)
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_multi_gpu.sh --config configs/your_config.yaml
```

### Method 2: Using Accelerate (Recommended)

Accelerate provides the most flexible and user-friendly interface for distributed training.

#### Setup Accelerate (One-time)

```bash
# Configure accelerate interactively
accelerate config

# Or use default multi-GPU config
accelerate config default
```

#### Launch Training

```bash
# Auto-detect GPUs and launch
accelerate launch src/lerobot/scripts/lerobot_train.py --config configs/your_config.yaml

# Specify number of GPUs
accelerate launch --num_processes=4 src/lerobot/scripts/lerobot_train.py --config configs/your_config.yaml

# With mixed precision (recommended for faster training)
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py --config configs/your_config.yaml
```

### Method 3: Using Native PyTorch DDP

For more control over the distributed training process, use the native DDP script.

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 src/lerobot/scripts/lerobot_train_ddp.py --config configs/your_config.yaml

# Using torch.distributed.launch (deprecated but still works)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/lerobot/scripts/lerobot_train_ddp.py --config configs/your_config.yaml

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    src/lerobot/scripts/lerobot_train_ddp.py --config configs/your_config.yaml
```

## Key Differences

| Feature | Accelerate | Native DDP |
|---------|-----------|------------|
| Ease of use | ⭐⭐⭐ Very easy | ⭐⭐ Moderate |
| Flexibility | ⭐⭐⭐ High | ⭐⭐ Medium |
| Mixed precision | Auto-handled | Manual setup |
| Gradient accumulation | Built-in | Manual |
| Multi-node support | Easy | Manual setup |
| DeepSpeed integration | Yes | No |
| FSDP support | Yes | No |

## Configuration Tips

### Batch Size

When using multi-GPU training, the **effective batch size** = `batch_size` × `num_gpus`

```yaml
# Example: 4 GPUs with batch_size=32
# Effective batch size = 32 × 4 = 128

batch_size: 32  # Per-GPU batch size
```

### Learning Rate Scaling

When increasing the effective batch size, you typically need to scale the learning rate:

```yaml
# Single GPU: batch_size=64, lr=1e-4
# 4 GPUs: batch_size=16 per GPU (effective 64), lr=1e-4 (same)
# 4 GPUs: batch_size=32 per GPU (effective 128), lr=2e-4 (scaled)
```

**Rule of thumb**: If you double the effective batch size, increase learning rate by √2 or 2×.

### Number of Workers

Adjust `num_workers` based on available CPU cores:

```yaml
# For 4 GPUs with 32 CPU cores
num_workers: 8  # 8 workers per GPU = 32 total workers
```

## Performance Optimization

### 1. Enable Mixed Precision Training

Mixed precision (FP16/BF16) can significantly speed up training:

```bash
# With Accelerate
accelerate launch --mixed_precision=bf16 src/lerobot/scripts/lerobot_train.py ...

# With native DDP (if implemented in config)
python src/lerobot/scripts/lerobot_train_ddp.py --use_amp ...
```

### 2. Use Gradient Accumulation

If GPU memory is limited, use gradient accumulation:

```yaml
# Simulate batch_size=128 with batch_size=32 and 4 accumulation steps
batch_size: 32
gradient_accumulation_steps: 4
```

### 3. Enable Efficient Backends

```yaml
# In your config or code
torch.backends.cudnn.benchmark = True  # Auto-select best convolution algorithm
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
```

## Monitoring Multi-GPU Training

### Check GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### WandB Logging

Only the main process (rank 0) logs to WandB to avoid duplicate entries:

```yaml
wandb:
  enable: true
  project: "my-robot-project"
```

## Troubleshooting

### Issue: "NCCL error: unhandled system error"

**Solution**: Check network configuration and firewall settings.

```bash
# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce per-GPU batch size
2. Enable gradient checkpointing (if supported by policy)
3. Use mixed precision training
4. Reduce number of workers

```yaml
batch_size: 16  # Reduce from 32
num_workers: 4  # Reduce from 8
```

### Issue: Uneven GPU Utilization

**Cause**: Imbalanced workload or data loading bottleneck

**Solutions**:
1. Ensure `DistributedSampler` is used
2. Increase `num_workers`
3. Check for data preprocessing bottlenecks

### Issue: Slower than Single GPU

**Causes**:
1. Communication overhead (small model/batch size)
2. Slow interconnect between GPUs
3. Data loading bottleneck

**Solutions**:
1. Increase batch size per GPU
2. Use gradient accumulation
3. Optimize data loading pipeline

## Advanced: Multi-Node Training

For training across multiple machines:

### Using Accelerate

```bash
# On each node, run:
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=0 \  # 0 for master, 1 for worker
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    src/lerobot/scripts/lerobot_train.py --config configs/your_config.yaml
```

### Using torchrun

```bash
# On master node (rank 0):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    src/lerobot/scripts/lerobot_train_ddp.py --config configs/your_config.yaml

# On worker node (rank 1):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    src/lerobot/scripts/lerobot_train_ddp.py --config configs/your_config.yaml
```

## Example: Training SmolVLA on Multiple GPUs

```bash
# Using 4 GPUs with Accelerate
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/lerobot/scripts/lerobot_train.py \
    --config configs/policy/smolvla.yaml \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=16 \
    wandb.enable=true \
    wandb.project=robot-learning

# Using 8 GPUs with native DDP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
    src/lerobot/scripts/lerobot_train_ddp.py \
    --config configs/policy/smolvla.yaml \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=8
```

## Performance Benchmarks

Example training speedup on NVIDIA A100 GPUs:

| GPUs | Batch Size | Steps/sec | Speedup | Efficiency |
|------|------------|-----------|---------|------------|
| 1    | 32         | 1.0       | 1.0×    | 100%       |
| 2    | 32         | 1.9       | 1.9×    | 95%        |
| 4    | 32         | 3.7       | 3.7×    | 93%        |
| 8    | 32         | 7.2       | 7.2×    | 90%        |

*Note: Actual performance depends on model size, batch size, and hardware.*

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
