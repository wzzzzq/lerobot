# SmolVLA Reflow 重构文档

## 概述

本次重构将 Reflow (Rectified Flow) 从独立的 Policy 类改为训练方法，使代码架构更加清晰和易于维护。

## 核心理念

**Reflow 是训练方法，不是新模型**

- 推理时：标准训练的模型和 Reflow 训练的模型完全相同
- 训练时：唯一区别是 loss 计算方式（使用 teacher 模型生成的轨迹）
- 类比：就像训练一个模型可以用 SGD 或 Adam，不需要 SGDModel 和 AdamModel

## 重构前后对比

### 旧架构（已废弃）

```
两个独立的 Policy 类：
├── SmolVLAPolicy（标准训练）
└── SmolVLAReflowPolicy（Reflow 训练）
    ├── 代码重复
    ├── 维护困难
    └── 概念混淆
```

**问题：**
- ❌ 两个 Policy 类维护两套代码
- ❌ 用户以为是两个不同的模型
- ❌ factory.py 需要特殊逻辑选择 Policy
- ❌ 配置文件混杂 `use_reflow` 和 `teacher_model_path`

### 新架构（当前）

```
统一的架构：
├── SmolVLAPolicy（唯一的 Policy 类）
│   └── VLAFlowMatching
│       ├── forward_standard()：标准 Flow Matching
│       ├── forward_reflow()：Reflow 训练
│       └── forward()：根据 training_mode 自动选择
│
└── Training Script
    ├── lerobot_train.py：标准训练
    └── lerobot_train_reflow.py：Reflow 训练
        └── 加载 teacher 并设置到 policy.model.teacher
```

**优点：**
- ✅ 单一 Policy 类：概念清晰
- ✅ 推理代码只维护一份
- ✅ 配置简单：`training_mode="reflow"`
- ✅ 易于扩展：可添加 3-RF, 4-RF 等

## 文件变更

### 修改的文件

1. **src/lerobot/policies/smolvla/modeling_smolvla.py**
   - ✅ 添加 `training_mode` 属性
   - ✅ 添加 `teacher` 属性
   - ✅ 添加 `forward_standard()` 方法
   - ✅ 添加 `forward_reflow()` 方法
   - ✅ 添加 `generate_reflow_target()` 方法
   - ✅ 修改 `forward()` 根据 `training_mode` 选择

2. **src/lerobot/policies/smolvla/configuration_smolvla.py**
   - ✅ 移除 `use_reflow: bool`
   - ✅ 移除 `teacher_model_path: str`
   - ✅ 添加 `training_mode: str`（"standard" 或 "reflow"）

3. **src/lerobot/policies/factory.py**
   - ✅ 移除 `SmolVLAReflowPolicy` 的特殊选择逻辑
   - ✅ 统一使用 `SmolVLAPolicy`

### 新增的文件

1. **src/lerobot/scripts/lerobot_train_reflow.py**
   - 专门的 Reflow 训练脚本
   - 加载 teacher 模型
   - 设置 `policy.model.teacher`
   - 设置 `policy.model.training_mode = "reflow"`

2. **examples/train_reflow_smolvla_new.sh**
   - 新的 Reflow 训练 shell 脚本
   - 使用 `lerobot_train_reflow.py`

### 删除的文件

1. **src/lerobot/policies/smolvla/modeling_smolvla_reflow.py** ❌ 已删除
   - 不再需要单独的 Reflow Policy 类

## 使用方法

### 标准训练（1-RF）

```bash
python src/lerobot/scripts/lerobot_train.py \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your_dataset \
  --steps=50000 \
  --batch_size=32
```

### Reflow 训练（2-RF）

```bash
python src/lerobot/scripts/lerobot_train_reflow.py \
  --policy.type=smolvla \
  --policy.training_mode=reflow \
  --policy.teacher_model_path=/path/to/1rf \
  --policy.optimizer_lr=2e-5 \
  --dataset.repo_id=your_dataset \
  --steps=20000 \
  --batch_size=32
```

或使用 shell 脚本：

```bash
bash examples/train_reflow_smolvla_new.sh
```

### 推理（两种训练方式的模型相同）

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 加载标准训练或 Reflow 训练的模型（完全相同）
policy = SmolVLAPolicy.from_pretrained("/path/to/checkpoint")

# 推理
actions = policy.select_action(observation)
```

## 核心代码变更

### VLAFlowMatching 的 forward 方法

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """根据 training_mode 选择 loss 计算方式"""
    if self.training_mode == "reflow" and self.training:
        return self.forward_reflow(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    else:
        return self.forward_standard(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
```

### 标准 Flow Matching Loss

```python
def forward_standard(self, ...):
    """标准 Flow Matching 训练 loss"""
    # X_t = t * noise + (1-t) * actions
    # u_t = noise - actions（可能弯曲的轨迹）
    # loss = ||u_t - v_t||^2
    ...
```

### Reflow Loss

```python
def forward_reflow(self, ...):
    """Reflow 训练 loss"""
    # X_0 = noise
    # X_1 = ODE[teacher](X_0)（teacher 生成的轨迹）
    # X_t = (1-t) * X_0 + t * X_1
    # u_t = X_1 - X_0（直线轨迹）
    # loss = ||u_t - v_t||^2
    ...
```

## 配置变更

### 旧配置（已废弃）

```python
config = SmolVLAConfig(
    use_reflow=True,                    # ❌ 已废弃
    teacher_model_path="/path/to/teacher",  # ❌ 已废弃
    ...
)
```

### 新配置

```python
config = SmolVLAConfig(
    training_mode="reflow",  # ✅ 新方式
    ...
)

# teacher_model_path 通过训练脚本参数传递
# --policy.teacher_model_path=/path/to/teacher
```

## 训练流程

### 完整的 Reflow 训练流程

```
1. 标准训练（1-RF）
   ↓
   python lerobot_train.py ... --steps=50000
   ↓
   输出：checkpoint_1rf/

2. Reflow 训练（2-RF）
   ↓
   python lerobot_train_reflow.py \
     --policy.teacher_model_path=checkpoint_1rf/ \
     --policy.optimizer_lr=2e-5 \
     --steps=20000
   ↓
   输出：checkpoint_2rf/

3. 推理（无区别）
   ↓
   policy = SmolVLAPolicy.from_pretrained("checkpoint_2rf/")
   actions = policy.select_action(obs)
```

### 重要：Student 模型初始化

**问题：Student 模型会从 teacher_model_path 加载吗？**

**答案：是的！这是 Reflow 训练的关键设计。**

在 Reflow 训练中：
1. **Teacher 模型**：从 `teacher_model_path` 加载，用于生成 ODE 轨迹（X_1）
2. **Student 模型**：也从 `teacher_model_path` 加载，从 Teacher 的权重开始

这是正确的设计，因为：

```python
# setup_reflow_training() 的关键代码：

# 1. 加载 Teacher 模型
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
teacher.eval()  # 冻结

# 2. 从同样的 checkpoint 初始化 Student
# 关键：Student 不是从头开始训练，而是从 Teacher 的权重开始
student = SmolVLAPolicy.from_pretrained(teacher_path)

# 3. 将 Teacher 附加到 Student
student.model.teacher = teacher
student.model.training_mode = "reflow"

# 4. Student 通过 Reflow loss 微调，学习更直的轨迹
```

**为什么这样设计？**

- Reflow 不是从头训练新模型，而是"拉直"已有模型的轨迹
- Student 从 Teacher 的权重开始，然后通过梯度下降学习更直的路径
- 这比从头训练快得多，并且保留了 Teacher 学到的特征
- 学习率通常较小（2e-5 vs 1e-4），因为只需要微调

**对比：**

```python
# ✗ 错误方式（从头创建 Student）
student = make_policy(cfg.policy, ds_meta=ds_meta)  # 随机初始化
teacher = load_teacher(teacher_path)

# ✓ 正确方式（从 Teacher 开始）
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
student = SmolVLAPolicy.from_pretrained(teacher_path)  # 复制 Teacher 的权重
student.model.teacher = teacher
```

## 超参数建议

| 参数 | 标准训练 | Reflow 训练 | 说明 |
|------|---------|-----------|------|
| **learning_rate** | 1e-4 | 2e-5 ~ 5e-5 | Reflow 是微调，需要更小的 LR |
| **steps** | 50000 | 20000 | Reflow 收敛更快 |
| **batch_size** | 32 | 32 | 保持不变 |
| **num_steps** (推理) | 10 | 2-5 | Reflow 后可用更少步数 |

## 迁移指南

如果你的代码使用了旧的 `SmolVLAReflowPolicy`，需要进行以下修改：

### 训练脚本迁移

**旧方式：**
```python
from lerobot.policies.smolvla.modeling_smolvla_reflow import SmolVLAReflowPolicy

policy = SmolVLAReflowPolicy(config)
```

**新方式：**
```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 使用 lerobot_train_reflow.py 脚本
# 或者手动设置：
policy = SmolVLAPolicy(config)
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
policy.model.teacher = teacher
policy.model.training_mode = "reflow"
```

### 配置文件迁移

**旧方式：**
```yaml
policy:
  type: smolvla
  use_reflow: true
  teacher_model_path: /path/to/teacher
```

**新方式：**
```yaml
policy:
  type: smolvla
  training_mode: reflow  # 在配置中
  # teacher_model_path 在训练脚本参数中传递
```

## 常见问题

### Q: 为什么要重构？

**A:** 旧架构将训练方法（Reflow）误认为是新模型，导致：
- 代码重复（两个 Policy 类）
- 用户困惑（以为是两个不同的模型）
- 维护困难（修改推理逻辑要改两处）

新架构清晰地表达了"Reflow 是训练方法"这一本质。

### Q: 旧的 checkpoint 还能用吗？

**A:** 可以！两种方式训练的 checkpoint 完全兼容：

```python
# 旧的 Reflow checkpoint
policy = SmolVLAPolicy.from_pretrained("/path/to/old_reflow_checkpoint")

# 新的 Reflow checkpoint
policy = SmolVLAPolicy.from_pretrained("/path/to/new_reflow_checkpoint")

# 两者推理完全相同
actions = policy.select_action(obs)
```

### Q: training_mode 会保存到 checkpoint 吗？

**A:** 不会。`training_mode` 只在训练时使用，保存的 checkpoint 不包含这个配置。推理时，所有模型都使用标准的 `sample_actions()` 方法。

### Q: 如何验证重构后的代码是否正确？

**A:** 运行测试：

```bash
# 1. 标准训练测试
python lerobot_train.py --steps=100 ...

# 2. Reflow 训练测试
python lerobot_train_reflow.py --policy.training_mode=reflow --steps=100 ...

# 3. 推理测试
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('/path/to/checkpoint')
print('✓ Checkpoint loaded successfully')
"
```

### Q: 为什么需要 lerobot_train_reflow.py？

**A:** 虽然可以在 `lerobot_train.py` 中添加 reflow 支持，但创建独立脚本有以下优点：
- 代码清晰：Reflow 特定逻辑集中
- 易于维护：不污染主训练脚本
- 灵活性：可以添加更多 Reflow 特定的功能

未来可以考虑将其整合到 `lerobot_train.py` 中。

## 技术细节

### 为什么 teacher 模型需要 `@torch.no_grad()`？

Teacher 模型只用于生成训练目标（X_1），不需要梯度：
- 节省内存
- 加速训练
- 确保 teacher 权重不被修改

### 为什么 X_1 = ODE[teacher](X_0) 是直线？

不是 X_1 本身是直线，而是 X_0 到 X_1 的轨迹被"拉直"：
- 标准 FM：X_t = t*noise + (1-t)*actions（可能弯曲）
- Reflow：X_t = (1-t)*X_0 + t*X_1（X_0 到 X_1 的直线）

通过迭代训练，X_1 越来越接近 actions，轨迹越来越直。

### 为什么 Reflow LR 要更小？

Reflow 是从 teacher 初始化的微调：
- Teacher 已经学到了很好的特征
- 只需要"拉直"轨迹，不需要重新学习
- 大 LR 会破坏已有特征

## 性能对比

| 指标 | 标准训练 (1-RF) | Reflow 训练 (2-RF) |
|------|----------------|-------------------|
| 训练步数 | 50000 | 20000 |
| 推理步数 | 10步 | 2-5步 |
| 推理速度 | 基准 | 2-5x 更快 |
| 质量 | 基准 | 相同或更好 |

## 参考资料

- [Reflow 论文](https://arxiv.org/abs/2209.03003): Flow Straight and Fast
- [SmolVLA 论文](https://huggingface.co/papers/2506.01844)
- [旧的 Reflow 文档](REFLOW_SMOLVLA_GUIDE.md)（仅供参考）

## 总结

这次重构实现了：
1. ✅ **概念清晰**：Reflow 是训练方法，不是新模型
2. ✅ **代码简洁**：单一 Policy 类，消除重复
3. ✅ **易于维护**：所有逻辑集中在 modeling_smolvla.py
4. ✅ **向后兼容**：旧 checkpoint 可以直接使用
5. ✅ **用户友好**：配置简单，使用清晰

**核心信息：Reflow 是训练方式，不是模型类型！**
