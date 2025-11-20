# Reflow Training - No VLM Sharing (Full Compatibility Mode)

## 更新 (2025-11-19)

**重要变更**: 移除了VLM sharing，确保reflow checkpoint与正常checkpoint完全一致。

## 工作原理

### 之前（使用VLM sharing）
```
训练时:
  Student Model: VLM (共享) + Expert (训练)
  Teacher Model: VLM (共享引用) + Expert (冻结)
  
优点: 节省GPU内存
缺点: Checkpoint可能存在兼容性问题
```

### 现在（无VLM sharing）
```
训练时:
  Student Model: VLM (独立，从teacher复制) + Expert (训练)  
  Teacher Model: VLM (独立) + Expert (冻结)
  
优点: Checkpoint与正常训练完全一致
缺点: 需要更多GPU内存（约2x VLM大小）
```

## Checkpoint兼容性

保存的reflow checkpoint现在与正常训练的checkpoint**完全相同**：

1. ✓ Config格式一致（`load_vlm_weights=true`）
2. ✓ 权重结构一致（完整的VLM + Expert）
3. ✓ 加载方式一致（标准`from_pretrained`）
4. ✓ 推理速度一致（无性能差异）

## 训练流程

### 1. Student初始化
```python
# __init__ 时，load_vlm_weights=false
# 不从HuggingFace加载VLM（避免重复下载）
self.model = VLAFlowMatching(config)
```

### 2. 从Teacher复制权重
```python
# _init_student_from_teacher()
teacher_policy = SmolVLAPolicy.from_pretrained(teacher_model_path)
self.load_state_dict(teacher_policy.state_dict(), strict=False)
# Student现在有完整的VLM + Expert权重
```

### 3. Teacher加载（无共享）
```python
# load_teacher_model() - 修改后
teacher = SmolVLAPolicy.from_pretrained(teacher_model_path)
# Teacher有自己独立的VLM + Expert
# 不再共享VLM引用
```

### 4. 保存Checkpoint
```python
# _save_pretrained()
# 保存所有student权重（包括完整VLM）
# Config中 load_vlm_weights=true
# 结果：与正常checkpoint完全一致
```

## 内存需求

### GPU内存估算
- VLM模型: ~1.5GB
- Expert模型: ~0.5GB
- **之前（VLM sharing）**: 1.5GB (VLM共享) + 0.5GB (Student Expert) + 0.5GB (Teacher Expert) = **2.5GB**
- **现在（无sharing）**: 1.5GB (Student VLM) + 1.5GB (Teacher VLM) + 0.5GB + 0.5GB = **4GB**

增加约**1.5GB** GPU内存需求。

## 训练配置

保持不变：
```bash
--policy.load_vlm_weights=false  # 训练时不从HF下载（从teacher复制）
--policy.use_reflow=true
--policy.teacher_model_path="..."
```

保存时自动修正为：
```json
{
  "load_vlm_weights": true,  // 推理时从checkpoint加载
  "use_reflow": false,       // 训练artifact已移除
  "teacher_model_path": null // 训练artifact已移除
}
```

## 验证Checkpoint一致性

```python
# 检查两个checkpoint是否一致
import safetensors.torch as st

normal_weights = st.load_file("normal_checkpoint/model.safetensors")
reflow_weights = st.load_file("reflow_checkpoint/model.safetensors")

# 应该有相同的keys
assert set(normal_weights.keys()) == set(reflow_weights.keys())

# 结构完全一致
print(f"Normal: {len(normal_weights)} weights")
print(f"Reflow: {len(reflow_weights)} weights")
```

## 相关修改

1. **`modeling_smolvla.py:load_teacher_model()`**
   - 移除VLM共享代码（第902-915行）
   - Teacher和Student使用独立VLM

2. **`modeling_smolvla.py:_save_pretrained()`**
   - 确保`load_vlm_weights=true`
   - 保存完整VLM权重

3. **`train_reflow_smolvla.sh`**
   - 更新注释说明无VLM sharing

## 旧文档（VLM Sharing版本）

以下是之前使用VLM sharing时的分析，保留作为参考：

---

# Reflow Checkpoint Config Fix (旧版本)

## 问题描述

Reflow训练出的checkpoint在推理时非常慢，每一步action之间间隔很久。

## 根本原因

保存的checkpoint中`config.json`的`load_vlm_weights`设置错误：

```json
{
  "load_vlm_weights": false,  // ❌ 错误！导致推理时VLM未加载预训练权重
  "use_reflow": false,        // ✓ 正确（训练artifact已移除）
  "teacher_model_path": null  // ✓ 正确（训练artifact已移除）
}
```

### 问题链路

1. **Reflow训练时**：
   - 设置`--policy.load_vlm_weights=false`（正确，因为VLM权重会从teacher复制）
   - 在`_init_student_from_teacher()`中加载teacher的完整权重（包括VLM）

2. **保存checkpoint时**（**BUG所在**）：
   - 调用`_save_pretrained()`
   - 移除`use_reflow`和`teacher_model_path`（正确）
   - **但保留了`load_vlm_weights=false`**（错误！）

3. **推理加载checkpoint时**：
   - 因为`load_vlm_weights=false`
   - `SmolVLMWithExpertModel.__init__`中不加载HuggingFace预训练VLM权重
   - 只创建随机初始化的VLM config
   - 导致推理时VLM层计算极慢

## 解决方案

### 1. 修复代码（已完成）

修改`src/lerobot/policies/smolvla/modeling_smolvla.py`中的`_save_pretrained`方法：

```python
def _save_pretrained(self, save_directory: Path) -> None:
    # Save config without reflow artifacts
    config_dict = {
        k: v for k, v in vars(self.config).items()
        if k not in ('use_reflow', 'teacher_model_path')
    }
    
    # Fix load_vlm_weights: if trained with reflow, VLM weights are included from teacher
    # So for inference, we should load them (they're in the saved weights)
    if self.config.use_reflow:
        config_dict['load_vlm_weights'] = True  # ✓ 修复！
    
    config_to_save = self.config.__class__(**config_dict)
    config_to_save._save_pretrained(save_directory)
    # ...
```

### 2. 修复现有checkpoint（已完成）

手动修复已保存的reflow checkpoint config：

```bash
# 修复checkpoint的config.json
python3 << 'EOF'
import json
config_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow/checkpoints/last/pretrained_model/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
config["load_vlm_weights"] = True
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
EOF
```

### 3. 训练脚本说明（已添加注释）

在`examples/train_reflow_smolvla.sh`中添加了说明：

```bash
# Note: --policy.load_vlm_weights=false is correct for reflow training
# The VLM weights will be loaded from the teacher model via VLM sharing
# When saving checkpoints, this will be automatically corrected to true
CUDA_VISIBLE_DEVICES=$GPU_ID python src/lerobot/scripts/lerobot_train.py \
  --policy.use_reflow=true \
  --policy.teacher_model_path="$TEACHER_MODEL_PATH" \
  --policy.load_vlm_weights=false \  # ✓ 训练时正确
  # ...
```

## VLM Sharing工作原理

Reflow训练使用VLM sharing来节省GPU内存：

1. **Teacher和Student共享同一个VLM**（只读）
2. **Teacher和Student各有独立的Expert层**（可训练）
3. VLM权重在`_init_student_from_teacher()`时从teacher复制到student

因此：
- **训练时**：`load_vlm_weights=false`（从teacher获取）
- **推理时**：`load_vlm_weights=true`（VLM权重已包含在checkpoint中）

## 验证

### 正确的config（推理时）

```json
{
  "load_vlm_weights": true,   // ✓ 加载VLM权重
  "num_steps": 10,
  "use_reflow": false,        // ✓ 已移除训练artifact
  "teacher_model_path": null  // ✓ 已移除训练artifact
}
```

### 推理速度对比

**修复前**（`load_vlm_weights=false`）：
- 每一步action之间间隔很久
- VLM使用随机初始化权重，计算低效

**修复后**（`load_vlm_weights=true`）：
- Action流畅，与正常checkpoint速度一致
- VLM使用从teacher继承的预训练权重

## 未来训练

使用修复后的代码训练新的reflow模型时，checkpoint会自动保存正确的config，不需要手动修复。

## 相关文件

- `src/lerobot/policies/smolvla/modeling_smolvla.py` - 修复`_save_pretrained`
- `examples/train_reflow_smolvla.sh` - 添加说明注释
- `src/lerobot/policies/smolvla/configuration_smolvla.py` - Config定义

## 技术细节

### `load_vlm_weights`的含义

- `True`: 从HuggingFace加载`vlm_model_name`的预训练权重
- `False`: 只加载config，不加载权重（随机初始化）

### Reflow checkpoint的正确状态

1. **包含VLM权重**（从teacher继承）
2. **包含Expert权重**（reflow训练得到）
3. **Config中`load_vlm_weights=true`**（推理时需要）
4. **Config中不包含训练artifact**（`use_reflow`, `teacher_model_path`）
