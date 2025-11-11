# SmolVLA Reflow 训练指南

## 目录

1. [什么是 Reflow？](#什么是-reflow)
2. [数学原理](#数学原理)
3. [使用方法](#使用方法)
4. [参数说明](#参数说明)
5. [示例](#示例)
6. [常见问题](#常见问题)

---

## 什么是 Reflow？

**Reflow (Rectified Flow)** 是一种用于"拉直"流匹配模型 ODE 轨迹的技术。通过迭代训练过程，将轨迹弯曲的模型转换为轨迹更直的模型。

### 核心概念

- **问题**：原始 Flow Matching 模型的概率流 ODE 轨迹是弯曲的
- **解决方案**：Reflow 训练新模型学习"拉直"的轨迹
- **优势**：
  - 轨迹更直 → 需要更少的 ODE 求解步骤
  - 更容易蒸馏为单步生成器
  - 提高采样效率和质量

### 为什么使用 Reflow？

1. **提高推理速度**：原始模型需要 10-50 步，Reflow 后可用 2-5 步达到相似质量
2. **提高动作质量**：更直的轨迹 → 更稳定的生成过程，减少累积误差
3. **适用于实时控制**：机器人应用需要低延迟，Reflow 显著减少推理时间

---

## 数学原理

### 标准 Flow Matching（SmolVLA 当前实现）

**轨迹定义**：
```
X_t = (1 - t) * noise + t * actions
```

**目标速度**：
```
u_t = noise - actions
```

**损失函数**：
```
L_FM = E[ ||u_t - v(X_t, t)||^2 ]
```

### Reflow（拉直轨迹）

Reflow 的关键是使用**教师模型**生成的轨迹作为新的训练目标。

#### Algorithm 1: 数据对生成

1. 采样噪声：`X_0 ~ N(0, 1)`
2. 使用教师模型 `v_k` 通过 ODE 求解生成 `X_1`：
   ```
   X_1 = ODE[v_k](X_0 | T) = X_0 + ∫_0^1 v_k(X_t, t | T) dt
   ```

#### 训练新模型 v_{k+1}

**轨迹定义**（线性插值）：
```
X_t = (1 - t) * X_0 + t * X_1
```

**直线速度**（目标）：
```
u_t = X_1 - X_0  (常数向量)
```

**Reflow 损失函数**（Equation 5）：
```
L_reflow = E_{X_0~N(0,1), T~D_T} [ ∫_0^1 ||(X_1 - X_0) - v_{k+1}(X_t, t|T)||^2 dt ]
```

其中：
- `v_k`：教师模型（已训练的模型）
- `v_{k+1}`：学生模型（正在训练的新模型）
- `X_0`：噪声起点
- `X_1`：教师模型生成的终点
- `(X_1 - X_0)`：直线速度（常数）

#### 核心区别

| 特性 | 标准 Flow Matching | Reflow |
|------|-------------------|--------|
| 目标速度 | `noise - actions` | `X_1 - X_0` |
| 终点 | 真实动作 (ground truth) | 教师模型生成的动作 |
| 轨迹形状 | 弯曲 | 更直 |
| ODE 步数 | 使用自己的 num_steps | 使用教师模型的 num_steps |

---

## 使用方法

### 前提条件

1. **已训练的 SmolVLA 模型**（作为教师模型）
2. **训练数据集**
3. **计算资源**（GPU，建议 A100 或类似）

### 快速开始

#### 1. 准备教师模型

```bash
# 选项 A: 使用 Hugging Face Hub 上的预训练模型
TEACHER_MODEL="lerobot/smolvla_base"

# 选项 B: 使用本地训练的模型
TEACHER_MODEL="/path/to/your/trained/smolvla/model"
```

#### 2. 启动 Reflow 训练

```bash
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path="lerobot/smolvla_base" \
  --dataset.repo_id="your_dataset" \
  --output_dir="outputs/smolvla_2rf" \
  --steps=50000
```

或使用提供的脚本：

```bash
# 修改 examples/train_reflow_smolvla.sh 中的配置
bash examples/train_reflow_smolvla.sh
```

---

## 参数说明

### Reflow 核心参数

#### `use_reflow` (bool, 默认: `false`)
- 是否启用 Reflow 训练模式
- `true`: 使用 Reflow 损失
- `false`: 使用标准 Flow Matching 损失

#### `teacher_model_path` (str, 必需当 `use_reflow=true`)
- 教师模型的路径
- 可以是 Hugging Face Hub 上的模型 ID（例如 `"lerobot/smolvla_base"`）
- 或本地文件系统路径（例如 `"outputs/smolvla_1rf"`）
- **注意**：ODE 求解将自动使用教师模型的 `num_steps` 配置

### 其他相关参数

#### `num_steps` (int, 默认: `10`)
- **学生模型**推理时的 ODE 求解步数
- Reflow 训练后，可以减少这个值来加速推理
- **训练时**：生成 X_1 使用教师模型的 `num_steps`

#### `optimizer_lr` (float, 默认: `1e-4`)
- Reflow 训练的学习率
- 可以与初始训练保持一致

#### `steps` (int)
- Reflow 训练的总步数
- 通常比初始训练少（50K-100K 步）

---

## 示例

### 使用配置文件

```yaml
# reflow_config.yaml
policy:
  type: smolvla
  use_reflow: true
  teacher_model_path: "lerobot/smolvla_base"
  num_steps: 10

dataset:
  repo_id: "your_dataset"
  batch_size: 32

training:
  steps: 50000
```

```bash
python lerobot/scripts/train.py --config reflow_config.yaml
```

### Python API 示例

```python
from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig

# 创建配置
config = SmolVLAConfig(
    use_reflow=True,
    teacher_model_path="lerobot/smolvla_base",
    freeze_vision_encoder=True,
    train_expert_only=True,
)

# 初始化模型
policy = SmolVLAPolicy(config)

# 训练时，会自动：
# 1. 加载教师模型
# 2. 使用教师模型的 num_steps 生成 X_1
# 3. 计算 Reflow 损失
```

### 训练流程

#### 阶段 1：基础训练（1-RF）
```bash
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --dataset.repo_id="your_dataset" \
  --output_dir="outputs/smolvla_1rf" \
  --steps=100000
```

#### 阶段 2：Reflow 训练（2-RF）
```bash
python lerobot/scripts/train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path="outputs/smolvla_1rf" \
  --dataset.repo_id="your_dataset" \
  --output_dir="outputs/smolvla_2rf" \
  --steps=50000
```

#### 阶段 3：评估比较
```bash
# 评估 1-RF（10 步）
python lerobot/scripts/eval.py \
  --policy.path="outputs/smolvla_1rf" \
  --policy.num_steps=10

# 评估 2-RF（5 步）
python lerobot/scripts/eval.py \
  --policy.path="outputs/smolvla_2rf" \
  --policy.num_steps=5

# 评估 2-RF（2 步）
python lerobot/scripts/eval.py \
  --policy.path="outputs/smolvla_2rf" \
  --policy.num_steps=2
```

---

## 常见问题

### Q1: Reflow 训练需要多长时间？

**A**:
- 1-RF（基础模型）：约 100K-200K 步，1-2 天（A100）
- 2-RF（Reflow）：约 50K-100K 步，0.5-1 天
- 训练时间会因需要生成 X_1 而略微增加（约 1.5-2x）

### Q2: 可以跳过基础训练直接做 Reflow 吗？

**A**: 不行。Reflow 需要一个已训练好的教师模型来生成 X_1。

### Q3: ODE 求解步数会影响训练吗？

**A**: 会。根据 Algorithm 1，生成 X_1 时使用教师模型的 `num_steps`。步数越多，X_1 越准确，但训练越慢。本实现自动使用教师模型的配置。

### Q4: Reflow 后性能会下降吗？

**A**: 通常不会。在大多数情况下：
- 2-RF 在**相同步数**下性能与 1-RF 相当或更好
- 2-RF 在**更少步数**下可以达到 1-RF（10 步）的性能
- 例如：2-RF (5 步) ≈ 1-RF (10 步)

### Q5: 可以迭代多次 Reflow 吗？

**A**: 可以，但收益递减：
- 1-RF → 2-RF：显著改进
- 2-RF → 3-RF：有改进，但较小
- 3-RF → 4-RF：改进很小

大多数情况下，2-RF 就足够了。

### Q6: 训练时 GPU 内存不足怎么办？

**A**:
- 减少 `batch_size`
- 使用 gradient checkpointing
- 使用更小的教师模型

### Q7: 如何验证 Reflow 是否有效？

**A**: 比较不同 `num_steps` 下的性能：
1. 评估 1-RF: num_steps = [2, 5, 10, 20]
2. 评估 2-RF: num_steps = [2, 5, 10, 20]
3. 2-RF 应该在更少步数下达到更好性能

---

## 技术细节

### 实现特点

1. **延迟加载教师模型**：只在第一次需要时加载
2. **动态生成 X_1**：每个批次实时生成，无需预生成数据
3. **梯度隔离**：教师模型使用 `@torch.no_grad()`
4. **自动配置 ODE 步数**：使用 `teacher.model.config.num_steps`

### 代码结构

```
modeling_smolvla.py
├── load_teacher_model()          # 延迟加载教师模型
├── generate_reflow_target()      # 使用教师模型生成 X_1
│   └── 使用 teacher.model.config.num_steps 进行 ODE 求解
├── forward_reflow()              # Reflow 训练 forward pass
└── forward()                     # 根据 use_reflow 选择模式
```

---

## 参考文献

1. **Rectified Flow**: [Liu et al., ICLR 2023](https://arxiv.org/abs/2209.03003)
2. **InstaFlow**: [Liu et al., ICLR 2024](https://arxiv.org/abs/2309.06380)
3. **SmolVLA**: [Hugging Face, 2024](https://huggingface.co/papers/2506.01844)
4. **Flow Matching**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)

---

## 总结

Reflow 实现了 Algorithm 1 中描述的轨迹拉直方法：

```
Algorithm 1: Training Text-Conditioned Rectified Flow
1. Input: Pre-trained model v_k (teacher)
2. Initialize v_{k+1} from v_k
3. For each training batch:
   a. Sample X_0 ~ N(0, 1)
   b. Generate X_1 = ODE[v_k](X_0) using teacher's num_steps
   c. Train v_{k+1} to predict straight-line velocity (X_1 - X_0)
4. Output: v_{k+1} (2-Rectified Flow)
```

**预期效果**：
- ✅ 2-5x 推理加速
- ✅ 保持或提高质量
- ✅ 为后续蒸馏奠定基础

如有问题或建议，请提交 GitHub Issue。
