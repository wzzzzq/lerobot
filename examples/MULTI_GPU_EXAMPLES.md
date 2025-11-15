# Multi-GPU Training Examples

本目录包含多卡训练的示例脚本。

## 📁 脚本说明

### 1. `train_smolvla_multi_gpu.sh`

针对 SmolVLA 模型的多卡训练示例，包含完整的配置。

**特点**：
- 预配置的 SmolVLA 训练参数
- 4 GPU 配置
- BF16 混合精度
- WandB 日志记录

**使用方法**：
```bash
# 修改脚本中的配置（数据集路径、输出目录等）
vim examples/train_smolvla_multi_gpu.sh

# 运行训练
bash examples/train_smolvla_multi_gpu.sh
```

### 2. `train_multi_gpu_template.sh`

通用多卡训练模板，可用于任何策略和数据集。

**特点**：
- 详细的配置注释
- 灵活的参数配置
- 适用于所有策略类型
- 包含错误检查和配置显示

**使用方法**：
```bash
# 1. 复制模板
cp examples/train_multi_gpu_template.sh examples/my_training.sh

# 2. 修改配置
vim examples/my_training.sh
# 主要修改：
#   - POLICY_PATH: 策略类型或预训练模型路径
#   - DATASET_REPO_ID: 你的数据集 ID
#   - NUM_GPUS: GPU 数量
#   - BATCH_SIZE: 每个 GPU 的 batch size

# 3. 运行训练
bash examples/my_training.sh
```

## ⚙️ 关键配置说明

### GPU 配置

```bash
# GPU 数量
NUM_GPUS=4

# 指定使用哪些 GPU（可选）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 混合精度模式
MIXED_PRECISION="bf16"  # 推荐用于 A100, RTX 3090+
# MIXED_PRECISION="fp16"  # 用于老显卡如 V100
# MIXED_PRECISION="no"    # 禁用混合精度
```

### Batch Size

```bash
# 每个 GPU 的 batch size
BATCH_SIZE=16

# 有效 batch size = BATCH_SIZE × NUM_GPUS
# 例如：16 × 4 = 64
```

**重要**：如果从单 GPU 迁移到多 GPU：
- 单 GPU: `batch_size=64` → 4 GPU: `batch_size=16` (保持有效 batch size = 64)

### Learning Rate

脚本使用策略的默认学习率。如果改变有效 batch size，可能需要调整：

```bash
# 在脚本中添加
--optimizer.lr=2e-4  # 根据有效 batch size 调整
```

### Workers 数量

```bash
# 每个 GPU 的 data loading workers
NUM_WORKERS=8

# 推荐值：4-8 workers per GPU
# 总 workers = NUM_WORKERS × NUM_GPUS
```

## 📊 使用场景示例

### 场景 1: 训练 SmolVLA（4 GPU）

```bash
# 使用 train_smolvla_multi_gpu.sh
# 修改数据集路径和输出目录后运行
bash examples/train_smolvla_multi_gpu.sh
```

### 场景 2: 训练 ACT Policy（2 GPU）

```bash
# 复制模板
cp examples/train_multi_gpu_template.sh examples/train_act_2gpu.sh

# 修改配置
# POLICY_PATH="act"
# NUM_GPUS=2
# BATCH_SIZE=32
# DATASET_REPO_ID="your-username/your-aloha-dataset"

# 运行
bash examples/train_act_2gpu.sh
```

### 场景 3: 使用特定 GPU

```bash
# 只使用 GPU 2 和 3
export CUDA_VISIBLE_DEVICES=2,3

# 修改脚本中的 NUM_GPUS=2
bash examples/train_multi_gpu_template.sh
```

### 场景 4: 从 Checkpoint 恢复训练

```bash
# 在脚本中添加
--resume=true \
--checkpoint_path=outputs/20250115_123456/checkpoints/last
```

## 🔍 监控训练

### 实时查看 GPU 使用

```bash
# 方法 1: nvidia-smi
watch -n 1 nvidia-smi

# 方法 2: gpustat（需要安装）
pip install gpustat
gpustat -i 1
```

### WandB 监控

脚本已配置 WandB 日志记录。训练开始后，查看链接：
```
https://wandb.ai/your-entity/your-project
```

## 🐛 常见问题

### 1. 显存不足 (OOM)

**解决方案**：
```bash
# 减小每个 GPU 的 batch size
BATCH_SIZE=8  # 从 16 降到 8

# 或减少 workers
NUM_WORKERS=4  # 从 8 降到 4
```

### 2. NCCL 错误

**解决方案**：在脚本中添加
```bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG=INFO
```

### 3. GPU 利用率低

**检查**：
```bash
# 查看 GPU 利用率
nvidia-smi

# 如果 < 80%，尝试：
# 1. 增大 BATCH_SIZE
# 2. 增大 NUM_WORKERS
# 3. 检查数据加载是否是瓶颈
```

### 4. 训练速度没有提升

**可能原因**：
- Batch size 太小（增大 BATCH_SIZE）
- Workers 不足（增大 NUM_WORKERS）
- 网络通信瓶颈（检查 GPU 互联）

## 📈 性能优化建议

### 1. 开启混合精度

```bash
MIXED_PRECISION="bf16"  # 可以获得 1.5-2× 加速
```

### 2. 优化 Batch Size

```bash
# 找到最大可用的 batch size（不 OOM）
# 从小开始测试：8, 16, 24, 32...
BATCH_SIZE=24  # 根据显存调整
```

### 3. 调整 Workers

```bash
# CPU 核心充足时
NUM_WORKERS=8  # 每 GPU 8 个 workers

# CPU 核心有限时
NUM_WORKERS=4  # 每 GPU 4 个 workers
```

### 4. 使用更快的存储

如果数据集在慢速存储上，考虑：
- 将数据集复制到本地 SSD
- 使用更快的网络存储
- 增加 prefetch_factor

## 🎯 最佳实践

1. **保持有效 batch size 一致**
   ```bash
   # 单 GPU: batch_size=64
   # 4 GPU: batch_size=16 (有效 = 64)
   ```

2. **从小规模开始测试**
   ```bash
   # 先用少量 steps 测试配置
   STEPS=100
   # 确认无误后再进行完整训练
   ```

3. **监控所有 GPU**
   ```bash
   # 确保所有 GPU 利用率相近
   watch -n 1 nvidia-smi
   ```

4. **使用 WandB 追踪实验**
   ```bash
   ENABLE_WANDB=true
   WANDB_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
   ```

5. **定期保存 checkpoint**
   ```bash
   --save_freq=1000  # 每 1000 步保存一次
   ```

## 📚 更多资源

- [完整多卡训练指南](../docs/MULTI_GPU_TRAINING.md)
- [快速入门指南](../MULTI_GPU_QUICKSTART.md)
- [Accelerate 文档](https://huggingface.co/docs/accelerate)

## 💡 提示

- 修改脚本前先备份：`cp script.sh script.sh.bak`
- 使用 `set -e` 确保出错时脚本停止
- 使用有意义的 `RUN_ID` 便于追踪实验
- 定期检查输出目录的磁盘空间
