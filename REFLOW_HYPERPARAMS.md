# Reflow Training Hyperparameters Guide

## Learning Rate设置

### 为什么Reflow需要更小的Learning Rate？

#### 1. 训练目标不同

**正常训练（Flow Matching）**:
- 从**随机初始化**开始
- 学习从noise到action的映射
- 需要较大LR来快速收敛
- 默认LR: `1e-4`

**Reflow训练（Rectified Flow）**:
- 从**已训练好的teacher**开始
- 不是从头学习，而是"**straightening**"已有的轨迹
- 模型已经知道如何生成actions
- 只需微调，让路径更直
- 建议LR: `1e-5` 到 `5e-5` (5-10x更小)

#### 2. Reflow的数学原理

Reflow的目标是让flow变直：

```
正常Flow Matching:
  学习: v(X_t, t) 使得 X_0 → X_1
  轨迹: 可能是弯曲的

Reflow (k+1轮):
  数据对: (X_0, X_1) where X_1 = ODE[v_k](X_0)
  学习: v_{k+1} 使得轨迹更直
  理想: v_{k+1} ≈ (X_1 - X_0) (直线)
```

因为teacher已经能生成正确的X_1，student只需要：
- **保持生成能力**（不要忘记已学的）
- **微调轨迹**（让路径更直）

用大LR会：
- ❌ 破坏已学的特征
- ❌ 导致训练不稳定
- ❌ 可能降低性能

#### 3. 实验建议（从Reflow论文）

| Training Stage | Recommended LR | Ratio to Normal |
|----------------|----------------|-----------------|
| Normal (k=0) | 1e-4 | 1x (baseline) |
| Reflow 1st (k=1) | 5e-5 | 0.5x |
| Reflow 2nd (k=2) | 2.5e-5 | 0.25x |
| Reflow 3rd+ | 1e-5 | 0.1x |

**解释**：
- 每一轮reflow，flow已经越来越直
- 需要的调整越来越小
- LR应该递减

## 推荐配置

### 第一次Reflow训练（从正常checkpoint）

```bash
# 保守策略（推荐）
REFLOW_LR="5e-5"  # 5x smaller

# 或者更激进（如果teacher很好）
REFLOW_LR="2e-5"  # 5x smaller
```

### 第二次Reflow训练（从第一次reflow checkpoint）

```bash
REFLOW_LR="2e-5"  # 10x smaller than original
```

### 其他Hyperparameters

```bash
# Batch size: 保持不变
BATCH_SIZE=32

# Training steps: 可以减少
# - 正常训练: 30k-50k steps
# - Reflow: 10k-20k steps (已经很好的初始化)
STEPS=20000

# Warmup: 减少或去除
# - 正常训练: 1000 steps
# - Reflow: 100-500 steps
WARMUP_STEPS=500

# Gradient clip: 保持或略微减小
GRAD_CLIP=10.0  # 或 5.0
```

## 训练脚本

### 自动设置（推荐）

已在`train_reflow_smolvla.sh`中设置：

```bash
# 在脚本中定义
REFLOW_LR="5e-5"

# 传递给训练命令
--policy.optimizer_lr="$REFLOW_LR"
```

### 手动调整

如果需要尝试不同的LR：

```bash
# 方法1: 修改脚本中的REFLOW_LR变量
REFLOW_LR="2e-5"  # 更小更保守

# 方法2: 命令行覆盖
bash train_reflow_smolvla.sh --policy.optimizer_lr=2e-5
```

## 监控训练

### 在WandB中检查

1. **Loss曲线**
   - 应该从较低值开始（teacher已经很好）
   - 平滑下降
   - 如果震荡 → LR太大

2. **Learning Rate**
   - 确认实际使用的LR
   - 检查scheduler是否正确

3. **对比teacher**
   - Student loss应该逐渐低于或接近teacher
   - 如果loss上升 → LR太大或训练有问题

### Warning Signs

**LR太大的症状**：
- ❌ Loss震荡
- ❌ Loss不下降或上升
- ❌ 梯度爆炸
- ❌ 生成的actions不稳定

**LR太小的症状**：
- ⚠️ Loss下降太慢
- ⚠️ 需要更多steps才能收敛
- ⚠️ 但不会破坏模型（更安全）

**建议**: 宁可LR小一点，也不要太大。Reflow的关键是稳定性。

## 理论背景

### Reflow算法（简化）

```python
# Reflow training (k+1 round)
for batch in data:
    # 1. Sample noise
    X_0 = sample_noise()
    
    # 2. Generate X_1 using teacher (k-th model)
    X_1 = ODE_solve(teacher, X_0)
    
    # 3. Sample timestep
    t = sample_uniform(0, 1)
    
    # 4. Linear interpolation
    X_t = (1-t) * X_0 + t * X_1
    
    # 5. Target velocity (straight line!)
    u_t = X_1 - X_0  # Not noise-actions anymore
    
    # 6. Student predicts velocity
    v_t = student(X_t, t)
    
    # 7. Loss (should be small if student ≈ teacher)
    loss = ||u_t - v_t||^2
    
    # 8. Update student with SMALL LR
    student.backward(loss, lr=SMALL_LR)
```

关键：
- `u_t = X_1 - X_0` 是直线
- Student已经接近teacher（初始化时复制的）
- 只需要小调整 → **小LR**

### 为什么不用Normal的LR？

使用正常的LR（1e-4）会：

```
Step 1: student ≈ teacher (很好)
Step 2: 大LR更新 → student偏离teacher
Step 3: Loss上升！
Step 4: 继续训练 → 可能收敛到更差的解
```

使用小LR（5e-5）：

```
Step 1: student ≈ teacher (很好)
Step 2: 小LR微调 → student略微改进
Step 3: Loss缓慢下降 ✓
Step 4: 继续训练 → 逐渐straighten flow
```

## 实际建议

### 第一次reflow训练

```bash
# 保守但安全（推荐新手）
REFLOW_LR="2e-5"
STEPS=20000

# 标准设置（推荐）
REFLOW_LR="5e-5"
STEPS=15000

# 激进设置（teacher质量很高时）
REFLOW_LR="7e-5"
STEPS=10000
```

### 调试策略

如果不确定用哪个LR：

1. **先用小的**（2e-5）
2. **观察100-500 steps**
3. **检查loss**：
   - 如果下降太慢 → 尝试5e-5
   - 如果震荡 → 降到1e-5
4. **重新训练**用最优LR

### Grid Search（可选）

如果有GPU资源，可以尝试：

```bash
# 测试不同LR
for lr in 1e-5 2e-5 5e-5 7e-5 1e-4; do
    echo "Testing LR=$lr"
    # 训练1000 steps
    # 比较loss
done
```

## 总结

| Parameter | Normal Training | Reflow Training |
|-----------|----------------|-----------------|
| LR | 1e-4 | 5e-5 (推荐) |
| Warmup | 1000 steps | 500 steps |
| Total Steps | 30k-50k | 15k-20k |
| Batch Size | 32 | 32 (相同) |

**记住**: Reflow是fine-tuning，不是从头训练！
