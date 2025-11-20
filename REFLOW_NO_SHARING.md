# Reflow Training - 移除VLM Sharing

## 修改摘要

为了确保reflow checkpoint与正常checkpoint完全一致，已移除VLM sharing机制。

## 关键修改

### 1. `src/lerobot/policies/smolvla/modeling_smolvla.py`

#### 修改A: `load_teacher_model()` - 禁用VLM sharing
```python
# 之前：
# Share VLM between teacher and student to save GPU memory
teacher_vlm = self._teacher_model.model.vlm_with_expert.vlm
student_vlm = self.vlm_with_expert.vlm
self._teacher_model.model.vlm_with_expert.vlm = student_vlm

# 现在：
# Note: VLM sharing is DISABLED to ensure checkpoint compatibility
# Both teacher and student have independent VLMs
print("[Reflow] Teacher and student use independent VLMs (no sharing)")
```

**影响**：
- 增加GPU内存使用（约+1.5GB）
- Teacher和Student各有独立VLM
- Checkpoint结构与正常训练完全一致

#### 修改B: `_save_pretrained()` - 确保config正确
```python
# 保存config时强制设置 load_vlm_weights=True
config_dict['load_vlm_weights'] = True
```

**影响**：
- 保存的config与正常checkpoint一致
- 推理时VLM从checkpoint正确加载

### 2. `examples/train_reflow_smolvla.sh`

更新注释说明：
```bash
# Note: Reflow training without VLM sharing (for full checkpoint compatibility)
# - Student model will have its own independent VLM
# - Teacher model will also have its own independent VLM  
# - This requires more GPU memory but ensures checkpoint is identical to normal training
```

## 训练流程（修改后）

```
1. 初始化Student Model
   ├─ load_vlm_weights=false (不从HF下载)
   └─ VLM随机初始化（临时）

2. _init_student_from_teacher()
   ├─ 加载teacher checkpoint
   ├─ 复制所有权重到student（包括VLM）
   └─ Student现在有完整VLM权重

3. load_teacher_model() 
   ├─ 加载独立的teacher model
   ├─ Teacher有自己的VLM（不共享）
   └─ 两个模型完全独立

4. 训练
   ├─ Student: VLM冻结，Expert训练
   └─ Teacher: 完全冻结，生成target

5. _save_pretrained()
   ├─ 过滤teacher model权重
   ├─ 保存student所有权重（VLM + Expert）
   └─ Config设置 load_vlm_weights=true

6. 推理
   ├─ from_pretrained() 加载checkpoint
   ├─ load_vlm_weights=true → 从checkpoint加载VLM
   └─ 与正常checkpoint完全相同
```

## Checkpoint对比

### 正常训练
```json
{
  "load_vlm_weights": true,
  "use_reflow": false,
  "teacher_model_path": null
}
```
权重：VLM + Expert（从头训练）

### Reflow训练（修改后）
```json
{
  "load_vlm_weights": true,
  "use_reflow": false,        // 自动移除
  "teacher_model_path": null  // 自动移除
}
```
权重：VLM（从teacher继承）+ Expert（reflow训练）

**结果：两者结构完全一致！**

## 内存使用

### 训练时
- **之前（VLM sharing）**: ~2.5GB
  - 1个VLM（共享）+ 2个Expert
- **现在（无sharing）**: ~4GB  
  - 2个VLM（独立）+ 2个Expert

### 推理时
- 无变化：只加载student model

## 验证步骤

### 1. 检查checkpoint config
```bash
cat checkpoint/config.json | grep "load_vlm_weights"
# 应该输出: "load_vlm_weights": true
```

### 2. 检查checkpoint大小
```bash
ls -lh checkpoint/model.safetensors
# 应该约1.2GB（与正常checkpoint相似）
```

### 3. 检查权重keys
```python
import safetensors.torch as st
weights = st.load_file("checkpoint/model.safetensors")
print(f"Total keys: {len(weights)}")
# 应该约500个keys

# 检查VLM权重
vlm_keys = [k for k in weights.keys() if 'vlm' in k]
print(f"VLM keys: {len(vlm_keys)}")
# 应该约340个keys
```

### 4. 推理速度测试
正常checkpoint和reflow checkpoint应该有相同的推理速度。

## 如果仍然慢怎么办

如果reflow checkpoint推理仍然慢，问题可能在：

1. **Normalization不匹配**
   - 检查正常和reflow的normalization模式
   - 确保postprocessor正确denormalize

2. **Action数值异常**
   - 检查action是否有NaN/Inf
   - 检查action数值范围是否合理

3. **物理仿真问题**
   - Action可能导致仿真求解困难
   - 检查是卡在推理还是`scene.step()`

## 重新训练

使用修改后的代码重新训练reflow模型：
```bash
bash examples/train_reflow_smolvla.sh
```

新checkpoint将与正常checkpoint完全兼容。

## 回滚（如果需要）

如果GPU内存不足，可以通过git恢复VLM sharing版本：
```bash
git checkout <commit_hash> -- src/lerobot/policies/smolvla/modeling_smolvla.py
```
