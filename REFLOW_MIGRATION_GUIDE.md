# Reflow 训练脚本迁移指南

## 变更概述

重构后的 Reflow 实现更简洁高效，训练脚本需要做小幅调整。

## 需要移除的参数

以下参数在新结构中**不再需要**（自动处理）：

### ❌ 移除：`--policy.load_vlm_weights=false`
**原因**：SmolVLAReflowPolicy 自动处理
- Teacher 加载时自动加载 VLM weights
- Student 创建时自动设置 `load_vlm_weights=false`
- 无需手动指定

### ❌ 移除：`--policy.train_expert_only=true`
**原因**：SmolVLAReflowPolicy 在 `__init__` 中自动设置
- Reflow 训练必须冻结 VLM，只训练 action expert
- 这个逻辑已内置在 `SmolVLAReflowPolicy.__init__()` 中

## 保留的参数

以下参数**必须保留**：

### ✅ 必须：`--policy.use_reflow=true`
触发 factory.py 自动选择 `SmolVLAReflowPolicy`

### ✅ 必须：`--policy.teacher_model_path=/path/to/teacher`
指定 teacher model 路径

### ✅ 可选：`--policy.freeze_vision_encoder=true`
冻结 vision encoder（推荐保留）

### ✅ 可选：`--policy.optimizer_lr=2e-5`
Reflow 专用学习率（比标准训练低 5-10倍）

## 迁移示例

### 旧脚本（重构前）
```bash
python lerobot_train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path=/path/to/teacher \
  --policy.load_vlm_weights=false \          # ❌ 需要移除
  --policy.train_expert_only=true \          # ❌ 需要移除
  --policy.freeze_vision_encoder=true \
  --batch_size=32 \
  --steps=20000
```

### 新脚本（重构后）
```bash
python lerobot_train.py \
  --policy.type=smolvla \
  --policy.use_reflow=true \
  --policy.teacher_model_path=/path/to/teacher \
  --policy.freeze_vision_encoder=true \      # ✅ 保留
  --batch_size=32 \
  --steps=20000
```

## 自动化逻辑

新结构中的自动处理流程：

1. **factory.py** 检测 `use_reflow=true`
   ```python
   if cfg.use_reflow:
       policy_cls = SmolVLAReflowPolicy  # 自动切换
   ```

2. **SmolVLAReflowPolicy.__init__()** 自动执行：
   ```python
   # 自动设置 train_expert_only=True
   config.train_expert_only = True
   
   # 加载 teacher（VLM 只加载 1 次）
   self.teacher = SmolVLAPolicy.from_pretrained(teacher_path)
   
   # 创建 student（不加载 VLM）
   config.load_vlm_weights = False
   self.model = VLAFlowMatchingReflow(config, teacher=self.teacher)
   
   # 复制权重
   self.load_state_dict(self.teacher.state_dict())
   ```

3. **VLAFlowMatchingReflow** 直接使用传入的 teacher
   ```python
   def __init__(self, config, teacher):
       self.teacher = teacher  # 无需 cache，直接使用
   ```

## 预期日志变化

### 旧版日志（加载 VLM 2次）
```
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...  # Student
[Reflow] Initializing student model from teacher: /path/to/teacher
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...  # Teacher
[Reflow] Student model initialized with teacher weights
```

### 新版日志（加载 VLM 1次）
```
[Reflow] Loading teacher model from /path/to/teacher
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...  # 只有这 1 次！
[Reflow] ✓ Teacher loaded and frozen
[Reflow] Creating student model (copying weights from teacher)
[Reflow] ✓ Student initialized from teacher
[Reflow] Total parameters: 450,046,176
[Reflow] Trainable parameters: 99,880,992 (22.18%)
```

## 性能提升

- **启动速度**：减少 50%（VLM 只加载 1 次）
- **内存占用**：启动时节省约 500MB（不需要临时加载第二个 VLM）
- **代码清晰度**：Reflow 逻辑集中在单一文件，易于维护

## 兼容性

### 向后兼容
- 旧的 checkpoint 仍可加载
- 新训练的 checkpoint 可用标准 `SmolVLAPolicy.from_pretrained()` 加载
- 推理代码无需修改

### 验证方法
```bash
# 验证只加载 1 次 VLM
lerobot-train ... --policy.use_reflow=true 2>&1 | grep -c "Loading.*HuggingFace"
# 输出应该是 1

# 验证 checkpoint 兼容性
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('path/to/reflow/checkpoint')
print('✓ Checkpoint compatible with standard SmolVLAPolicy')
"
```

## 故障排查

### 如果出现 "ImportError: SmolVLAReflowPolicy"
**原因**：factory.py 未正确更新
**解决**：确保 `src/lerobot/policies/factory.py` 包含 reflow 检测逻辑

### 如果仍看到 VLM 加载 2 次
**原因**：使用了旧的 modeling_smolvla.py
**解决**：确保 modeling_smolvla.py 已还原到 783 行（无 reflow 代码）

### 如果出现 "teacher_model_path not found"
**原因**：路径错误或 checkpoint 结构不对
**解决**：检查路径是否指向包含 `config.json` 和 `model.safetensors` 的目录

## 已更新的脚本

以下脚本已适配新结构：

1. ✅ `train_2rf_with_teacher.sh` - 主训练脚本
2. ✅ `examples/train_reflow_smolvla.sh` - 详细版训练脚本

## 测试清单

迁移后建议测试：

- [ ] 启动训练，确认 VLM 只加载 1 次
- [ ] 检查日志，确认显示 `[Reflow] ✓ Student initialized from teacher`
- [ ] 验证参数统计，trainable params 约 22%
- [ ] 训练几个 step，确认 loss 正常
- [ ] 保存 checkpoint，用标准 SmolVLAPolicy 加载验证

## 相关文档

- **REFLOW_REFACTOR.md** - 完整重构说明
- **modeling_smolvla_reflow.py** - Reflow 实现代码
- **factory.py** - 自动选择逻辑

## 问题反馈

如遇到问题，请检查：
1. 是否移除了 `--policy.load_vlm_weights=false`
2. 是否移除了 `--policy.train_expert_only=true`
3. 是否保留了 `--policy.use_reflow=true`
4. 日志中 VLM 加载次数是否为 1
