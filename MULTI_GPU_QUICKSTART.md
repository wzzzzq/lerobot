# Multi-GPU Training Quick Start

LeRobot 现有的 `lerobot_train.py` 已经完整支持多卡训练，通过 [Accelerate](https://huggingface.co/docs/accelerate) 实现。

## 🚀 最快上手方式

```bash
# 1. 安装 accelerate（如果还没安装）
pip install accelerate

# 2. 配置 accelerate（一次性设置）
accelerate config default

# 3. 启动多卡训练（自动检测所有 GPU）
accelerate launch src/lerobot/scripts/lerobot_train.py \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=your-username/your-dataset \
    batch_size=16
```

就是这么简单！

## 📊 常用命令

### 指定 GPU 数量

```bash
# 使用 4 张 GPU
accelerate launch --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py [your args]

# 使用 2 张 GPU
accelerate launch --num_processes=2 \
    src/lerobot/scripts/lerobot_train.py [your args]
```

### 指定使用哪些 GPU

```bash
# 只使用 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 \
    src/lerobot/scripts/lerobot_train.py [your args]

# 使用 GPU 2,3,4,5
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py [your args]
```

### 开启混合精度（更快）

```bash
# 推荐：BF16（适用于 A100, RTX 3090 等）
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py [your args]

# 或使用 FP16（适用于老显卡）
accelerate launch \
    --num_processes=4 \
    --mixed_precision=fp16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py [your args]
```

## 🎯 训练 SmolVLA 示例（4 卡）

### 方法 1：使用示例脚本（最简单）

```bash
# 修改配置
vim examples/train_smolvla_multi_gpu.sh

# 运行训练
bash examples/train_smolvla_multi_gpu.sh
```

### 方法 2：直接使用 Accelerate

```bash
# 基础多卡训练
accelerate launch --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=christianwang-sjtu/so100-red-dustbin \
    batch_size=12 \
    output_dir=outputs/smolvla_4gpu

# 加上混合精度加速
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --multi_gpu \
    src/lerobot/scripts/lerobot_train.py \
    policy.pretrained_path=lerobot/smolvla_base \
    dataset.repo_id=christianwang-sjtu/so100-red-dustbin \
    batch_size=12 \
    output_dir=outputs/smolvla_4gpu_bf16
```

## 💡 重要配置说明

### Batch Size 调整

**有效 batch size** = `batch_size` × `GPU 数量`

```yaml
# 例如：4 卡，batch_size=12
# 有效 batch size = 12 × 4 = 48

# 如果你之前单卡用 batch_size=48
# 现在 4 卡应该用 batch_size=12（保持有效 batch size 一致）
batch_size: 12
```

### Learning Rate 调整

当有效 batch size 改变时，通常需要调整学习率：

```yaml
# 单卡：batch_size=48, lr=1e-4
# 4 卡：batch_size=12, lr=1e-4（有效 batch size 相同，lr 不变）

# 如果增大有效 batch size：
# 单卡：batch_size=48,  lr=1e-4
# 4 卡：batch_size=48,  lr=2e-4（有效 batch size = 192，lr 翻倍）
```

**经验规则**：有效 batch size 翻倍，lr 增加 √2 到 2 倍。

### Workers 数量

```yaml
# 假设 4 卡，32 核 CPU
num_workers: 8  # 每卡 8 个 worker，总共 32 个 worker

# 推荐范围：每卡 4-8 个 worker
```

## 🔍 监控训练

### 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat（更友好）
pip install gpustat
gpustat -i 1
```

### 期望的 GPU 利用率

- ✅ **良好**：所有 GPU 利用率 80-100%
- ⚠️ **一般**：利用率 50-80%（可能需要增大 batch_size）
- ❌ **不佳**：利用率 <50%（数据加载瓶颈或 batch_size 太小）

## 🔧 常见问题

### 显存不足（OOM）

**解决方案**：
1. 减小每卡的 `batch_size`
2. 开启混合精度 `--mixed_precision=bf16`
3. 减少 `num_workers`

```yaml
batch_size: 8   # 从 16 降到 8
num_workers: 4  # 从 8 降到 4
```

### 训练速度慢

**诊断**：
```bash
# 查看 GPU 利用率
nvidia-smi
```

**解决方案**：
```yaml
# 如果显存充足，增大 batch_size
batch_size: 24  # 从 12 增加到 24

# 增加 workers（如果 CPU 核心充足）
num_workers: 12  # 从 8 增加到 12
```

### GPU 之间负载不均

**检查**：
```bash
# 监控各个 GPU
nvidia-smi dmon -i 0,1,2,3
```

**原因**：通常是硬件差异或数据分布问题
**解决**：确保所有 GPU 型号相同

## 📈 性能参考

在 NVIDIA A100 上训练 SmolVLA 的加速比：

| GPU 数量 | 每卡 Batch Size | 加速比 | 效率 |
|---------|----------------|--------|------|
| 1       | 32             | 1.0×   | 100% |
| 2       | 32             | 1.9×   | 95%  |
| 4       | 32             | 3.7×   | 93%  |
| 8       | 32             | 5.4×   | 68%  |

*注：效率 = 加速比 / GPU数量，随着 GPU 数量增加，效率会因通信开销而降低*

## ⚠️ 重要提示

1. **有效 Batch Size**：N 卡 × batch_size = 有效 batch_size
2. **学习率调整**：改变有效 batch size 时需要调整 lr
3. **随机种子**：每张 GPU 使用不同种子（seed + rank）确保数据不重复
4. **Checkpoint 保存**：只有主进程（rank 0）保存模型
5. **WandB 日志**：只有主进程上传日志，避免重复

## 📖 完整文档

详细信息请查看：
- **示例脚本**: [`examples/MULTI_GPU_EXAMPLES.md`](examples/MULTI_GPU_EXAMPLES.md)
- **详细指南**: [`docs/MULTI_GPU_TRAINING.md`](docs/MULTI_GPU_TRAINING.md)

## ✅ 现有功能

`lerobot_train.py` 已经内置：

✅ 自动检测多 GPU（通过 Accelerate）
✅ 梯度跨 GPU 同步
✅ 混合精度训练
✅ 分布式数据加载
✅ Checkpoint 保存（只在主进程）
✅ WandB 日志（只在主进程）
✅ 指标跨进程聚合
✅ 多节点训练支持

**无需修改任何代码，直接使用 Accelerate 启动即可！**

## 🎓 下一步

1. 查看示例脚本：`examples/train_smolvla_multi_gpu.sh`
2. 阅读完整指南：`docs/MULTI_GPU_TRAINING.md`
3. 尝试不同的 batch size 和学习率组合
4. 使用混合精度加速训练
5. 用 WandB 监控训练过程

## 参考资料

- [Accelerate 文档](https://huggingface.co/docs/accelerate)
- [Accelerate 快速入门](https://huggingface.co/docs/accelerate/quicktour)
- [启动分布式训练](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
