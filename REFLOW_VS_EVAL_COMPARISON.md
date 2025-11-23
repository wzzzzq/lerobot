# Reflow Training vs RobotWin Eval 数据处理对比分析

## 问题描述
Velocity质量测试显示student模型预测误差只有4.5%（良好），但在robotwin eval时机械臂完全乱动。这表明训练和inference在数据处理上存在不一致。

## 详细对比

### 1. 图片处理流程

#### **Eval脚本** (`examples/robotwin_eval/eval_policy_smolvla.py`)
```python
# Line 63-67: prepare_img函数
def prepare_img(img):
    # Convert HWC to CHW, normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = img.astype(np.float32) / 255.0  # [0, 255] → [0, 1]
    return torch.from_numpy(img)

# Line 88: 传给preprocessor
self.observation_window = self.preprocessor(observation)
```

#### **Policy的prepare_images** (`src/lerobot/policies/smolvla/modeling_smolvla.py:344-384`)
```python
def prepare_images(self, batch):
    for key in present_img_keys:
        img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]

        # Resize with padding
        if self.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

        # ⚠️ 关键: 从[0,1]归一化到[-1,1] (SigLIP要求)
        img = img * 2.0 - 1.0  # [0, 1] → [-1, 1]
```

**流程**:
- Eval: 环境RGB [0,255] → prepare_img → [0,1] CHW → preprocessor → prepare_images → **[-1,1]**
- Training: Dataset已经是[0,1] CHW → prepare_images → **[-1,1]**

**✓ 这个应该是一致的**

---

### 2. 相机顺序

#### **Eval脚本**
```python
# Line 78-84: Camera names - MUST match training data order
camera_names = ["head_camera", "left_camera", "right_camera"]
for i, camera_name in enumerate(camera_names):
    if i < len(img_arr):
        key = f"observation.images.{camera_name}"
        observation[key] = prepare_img(img_arr[i])

# Line 182-186: encode_obs提取相机
input_rgb_arr = [
    observation["observation"]["head_camera"]["rgb"],  # 0
    observation["observation"]["left_camera"]["rgb"],   # 1
    observation["observation"]["right_camera"]["rgb"],  # 2
]
```

#### **Training Dataset**
需要检查数据集中相机的顺序！

**❓ 需要验证**: Dataset中的相机顺序是否和eval完全一致？
- Dataset keys: `observation.images.head_camera`, `observation.images.left_camera`, `observation.images.right_camera`
- Policy config中的 `image_features` 顺序是什么？

---

### 3. State处理

#### **Eval脚本**
```python
# Line 69: 直接转float32
state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))

# Line 73: 添加到observation
observation = {
    "observation.state": state_tensor,
    ...
}
```

#### **Policy的prepare_state** (`modeling_smolvla.py:413-417`)
```python
def prepare_state(self, batch):
    """Pad state"""
    state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
    state = pad_vector(state, self.config.max_state_dim)  # Pad到max_state_dim
    return state
```

**流程**:
- Eval: state (14-dim) → preprocessor (可能normalize) → prepare_state → **padded state**
- Training: batch["observation.state"] → prepare_state → **padded state**

**❓ 需要验证**:
1. Eval传入的state维度是否正确（14-dim）？
2. State是否需要归一化？Preprocessor的NormalizerProcessorStep会处理state吗？

---

### 4. Language处理

#### **Eval脚本**
```python
# Line 74: 使用instruction字符串
observation = {
    "task": self.instruction if isinstance(self.instruction, str) else self.instruction[0],
}

# Line 88: Preprocessor会tokenize
self.observation_window = self.preprocessor(observation)
```

#### **Training**
```python
# prepare_reflow_batch (lerobot_train_reflow.py:207-208)
lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]  # 已经tokenized
lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
```

#### **Preprocessor** (`processor_smolvla.py:73-78`)
```python
SmolVLANewLineProcessor(),  # 添加newline
TokenizerProcessorStep(
    tokenizer_name=config.vlm_model_name,
    padding=config.pad_language_to,
    padding_side="right",
    max_length=config.tokenizer_max_length,
),
```

**✓ 这个应该是一致的**: eval使用task string → preprocessor tokenize，training使用dataset的task → preprocessor tokenize

---

### 5. Inference方法对比

#### **Eval脚本** (`eval_policy_smolvla.py:101`)
```python
# select_action
action_tensor = self.policy.select_action(self.observation_window)
```

#### **select_action流程** (`modeling_smolvla.py:291-311`)
```python
def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    self.eval()
    batch = self._prepare_batch(batch)  # adapt_to_pi_aloha decode

    if len(self._queues[ACTION]) == 0:
        actions = self._get_action_chunk(batch, noise)
        # (batch_size, n_action_steps, action_dim)
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

    return self._queues[ACTION].popleft()
```

#### **_get_action_chunk** (`modeling_smolvla.py:248-272`)
```python
def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    # ⚠️ 使用sample_actions (ODE integration)
    actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

    # ⚠️⚠️⚠️ 关键: Unpad actions到original_action_dim (14)
    original_action_dim = self.config.action_feature.shape[0]
    actions = actions[:, :, :original_action_dim]  # 32 → 14

    if self.config.adapt_to_pi_aloha:
        actions = self._pi_aloha_encode_actions(actions)

    return actions
```

#### **Reflow Training** (`lerobot_train_reflow.py:223-229`)
```python
# Teacher生成X_0
X_0 = teacher.model.sample_actions(
    images, img_masks, lang_tokens, lang_masks, state, noise=X_1_padded
)
# CRITICAL FIX: Do NOT unpad X_0!
# 保持X_0为32-dim
```

**✓ 这里应该没问题**:
- Training: teacher生成32-dim X_0 → student学习velocity → 输出32-dim
- Inference: student生成32-dim → unpad到14-dim → 返回给环境

Unpad只是去掉padding维度（应该接近0），不影响前14维的值。

---

### 6. Preprocessor/Postprocessor使用

#### **Eval脚本** (`eval_policy_smolvla.py:165-169`)
```python
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=policy_path,  # 从checkpoint加载
)
```

#### **Preprocessor Pipeline** (`processor_smolvla.py:69-85`)
```python
input_steps = [
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),  # 添加batch维度
    SmolVLANewLineProcessor(),         # task添加\n
    TokenizerProcessorStep(...),       # tokenize task
    DeviceProcessorStep(device=config.device),  # 移到GPU
    NormalizerProcessorStep(          # ⚠️ 归一化state和action
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
    ),
]
```

**❓❓❓ 关键问题**:
1. **NormalizerProcessorStep会归一化state吗？**
   - 如果会，eval的state需要先归一化再传入
   - Training的state从dataset来，可能已经归一化了吗？

2. **Dataset返回的state是否已经归一化？**
   - 需要检查dataset加载时是否应用了normalization

---

### 7. adapt_to_pi_aloha处理

#### **Inference** (`modeling_smolvla.py:274-277, 269-270`)
```python
def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    if self.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
    return batch

# In _get_action_chunk:
if self.config.adapt_to_pi_aloha:
    actions = self._pi_aloha_encode_actions(actions)
```

#### **Training** (`lerobot_train_reflow.py:200-202`)
```python
if student.config.adapt_to_pi_aloha:
    batch[OBS_STATE] = student._pi_aloha_decode_state(batch[OBS_STATE])
    batch[ACTION] = student._pi_aloha_encode_actions_inv(batch[ACTION])
```

**✓ 这个应该是一致的**

---

## 🚨 关键可疑点总结

### 1. **State归一化** (最可疑!)
- **问题**: NormalizerProcessorStep可能会归一化state
- **影响**: 如果training时state已归一化，但eval时state是原始值，会导致完全错误的预测
- **验证方法**:
  1. 检查policy config中的`normalization_mapping`和`input_features`
  2. 打印eval时preprocessor前后的state值范围
  3. 对比training dataloader返回的state值范围

### 2. **相机顺序** (需要验证)
- **问题**: Policy config中的`image_features`顺序可能和eval不一致
- **影响**: 相机feed错位会导致模型看到错误的视角
- **验证方法**:
  1. 检查policy config的`image_features`列表顺序
  2. 检查dataset中相机的顺序
  3. 对比eval中camera_names列表

### 3. **Batch dimension** (preprocessor处理)
- **问题**: AddBatchDimensionProcessorStep会添加batch维度
- **影响**: 如果state是(14,)，会变成(1, 14)；但prepare_state期望(batch, 14)
- **验证方法**: 打印preprocessor输出的observation window的shapes

---

## 建议的Debug步骤

### Step 1: 创建对比脚本
创建一个脚本同时运行training的prepare_reflow_batch和eval的observation处理，对比：
- Images shape和value range
- State shape和value range
- Language tokens
- 最终传给model.sample_actions的所有inputs

### Step 2: 检查Config
打印policy.config中的：
- `normalization_mapping`
- `input_features`
- `image_features`的顺序
- `adapt_to_pi_aloha`
- `resize_imgs_with_padding`

### Step 3: 添加Debug日志到Eval
在eval脚本中添加日志：
```python
# After preprocessor
print("Preprocessed observation:")
for k, v in self.observation_window.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")

# Before sample_actions in _get_action_chunk
print("Inputs to sample_actions:")
print(f"  images: {len(images)} cameras, shape={images[0].shape}, range=[{images[0].min():.3f}, {images[0].max():.3f}]")
print(f"  state: shape={state.shape}, range=[{state.min():.3f}, {state.max():.3f}]")
print(f"  lang_tokens: shape={lang_tokens.shape}")
```

### Step 4: 对比Dataset vs Eval
从training dataset取一个batch，打印：
- batch["observation.state"]的shape和range
- batch["observation.images.xxx"]的shape和range
- 和eval中对应值对比

---

## 最可能的根本原因

基于代码分析，**最可能的问题是State归一化不一致**：

1. Training dataset可能返回归一化后的state (比如[-1, 1]或[0, 1])
2. Eval传入的是原始joint state (比如[-3.14, 3.14]等物理单位)
3. 即使preprocessor有NormalizerProcessorStep，它可能需要正确的stats才能归一化

**验证**: 对比training时batch["observation.state"]和eval时observation["observation.state"]的数值范围。
