# ALOHA 数据转换和 SmolVLA 训练指南

本指南提供详细的步骤来转换ALOHA HDF5数据集到LeRobot v3.0格式，并使用该数据集训练SmolVLA模型。

## 目录

- [环境设置](#环境设置)
- [数据转换](#数据转换)
- [训练SmolVLA](#训练smolvla)
- [常见问题](#常见问题)

---

## 环境设置

### 1. 创建并激活虚拟环境

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

### 2. 安装FFmpeg

```bash
conda install ffmpeg -c conda-forge
```

### 3. 安装LeRobot

从源码安装（推荐用于开发）：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 4. 安装SmolVLA依赖

```bash
pip install -e ".[smolvla]"
```

这会安装以下依赖：
- transformers>=4.53.0
- num2words
- accelerate>=1.7.0
- safetensors>=0.4.3

### 5. 配置Weights & Biases（可选）

如果你想使用W&B进行实验跟踪：

```bash
wandb login
```

---

## 数据转换

### 数据准备

在转换之前，你需要准备：

1. **HDF5文件目录**：包含ALOHA录制的episode文件（`episode0.hdf5`, `episode1.hdf5`等）
2. **指令文件目录**：包含每个episode的指令JSON文件

#### 指令文件格式

为每个episode创建一个JSON文件（例如`episode0.json`, `episode1.json`）：

```json
{
  "seen": [
    "pick up the red block",
    "grasp the red object",
    "take the red cube"
  ]
}
```

脚本会从`seen`列表中随机选择一条指令。

### 转换命令

#### 基本转换

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /path/to/hdf5/files \
    --instruction-dir /path/to/instructions \
    --repo-id username/my-aloha-dataset \
    --output-dir /path/to/output
```

#### 参数说明

- `--raw-dir`: HDF5文件所在目录（必需）
- `--instruction-dir`: 指令JSON文件所在目录（必需）
- `--repo-id`: 数据集标识符，格式：`username/dataset-name`（必需）
- `--output-dir`: 输出目录（可选，默认：`~/.cache/huggingface/lerobot/{repo-id}`）
- `--episodes`: 要转换的特定episode索引列表（可选，默认：全部）
- `--push-to-hub`: 是否上传到Hugging Face Hub（可选）

#### 示例

##### 转换所有episodes到自定义目录

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha_recordings \
    --instruction-dir ./data/instructions \
    --repo-id myusername/aloha-pick-place \
    --output-dir ./data/lerobot/aloha-pick-place
```

##### 转换特定episodes

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha_recordings \
    --instruction-dir ./data/instructions \
    --repo-id myusername/aloha-pick-place \
    --output-dir ./data/lerobot/aloha-pick-place \
    --episodes 0 1 2 3 4
```

##### 转换并上传到Hub

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha_recordings \
    --instruction-dir ./data/instructions \
    --repo-id myusername/aloha-pick-place \
    --push-to-hub
```

### 验证转换后的数据集

#### 使用可视化工具

```bash
lerobot-dataset-viz \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --mode local \
    --episode-index 0
```

这会打开Rerun.io界面，显示相机流、机器人状态和动作。

#### 加载数据集进行检查

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从本地加载
dataset = LeRobotDataset(
    repo_id="myusername/aloha-pick-place",
    root="./data/lerobot"
)

print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Total frames: {dataset.meta.total_frames}")
print(f"FPS: {dataset.meta.info['fps']}")
print(f"Features: {list(dataset.features.keys())}")

# 查看第一帧
sample = dataset[0]
for key in sample.keys():
    print(f"{key}: {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")
```

### 转换后的数据集结构

```
output_dir/
├── data/
│   └── chunk-000/
│       └── file-000.parquet          # 状态和动作数据
├── videos/
│   └── {camera_name}/
│       └── chunk-000/
│           └── file-000.mp4          # 视频数据
└── meta/
    ├── info.json                      # 数据集元信息
    ├── episodes/                      # Episode元数据
    ├── tasks/                         # 任务信息
    └── episodes_stats/                # Episode统计信息
```

---

## 训练SmolVLA

### 方法1: 使用命令行训练（推荐）

#### 基本训练命令

```bash
lerobot-train \
    --policy-name smolvla \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --output-dir outputs/smolvla_aloha \
    --num-train-iters 10000 \
    --batch-size 8 \
    --eval-freq 1000 \
    --save-freq 1000 \
    --log-freq 100
```

#### 参数说明

- `--policy-name`: 策略类型（这里使用`smolvla`）
- `--repo-id`: 数据集的repo ID
- `--root`: 数据集本地路径（如果使用本地数据集）
- `--output-dir`: 训练输出目录
- `--num-train-iters`: 训练迭代次数
- `--batch-size`: 批量大小
- `--eval-freq`: 评估频率
- `--save-freq`: 保存检查点频率
- `--log-freq`: 日志记录频率

#### 使用W&B进行实验跟踪

```bash
lerobot-train \
    --policy-name smolvla \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --output-dir outputs/smolvla_aloha \
    --num-train-iters 10000 \
    --batch-size 8 \
    --use-wandb \
    --wandb-project my-robotics-project \
    --wandb-run-name smolvla-aloha-run1
```

#### 多GPU训练

```bash
accelerate launch --multi_gpu --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py \
    --policy-name smolvla \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --output-dir outputs/smolvla_aloha \
    --num-train-iters 10000 \
    --batch-size 32
```

### 方法2: 使用Python脚本训练

创建训练脚本 `train_smolvla.py`：

```python
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

def main():
    # 配置
    output_directory = Path("outputs/train/smolvla_aloha")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 10000
    batch_size = 8
    learning_rate = 1e-4
    log_freq = 100

    # 加载数据集元数据
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="myusername/aloha-pick-place",
        root="./data/lerobot"
    )

    # 准备特征配置
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 创建策略配置
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features
    )

    # 初始化策略
    policy = SmolVLAPolicy(cfg)
    policy.train()
    policy.to(device)

    # 创建预处理器和后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        dataset_stats=dataset_metadata.stats
    )

    # 准备delta_timestamps（根据策略要求）
    delta_timestamps = {
        f"observation.images.{key}": [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
        for key in dataset_metadata.video_keys
    }
    delta_timestamps.update({
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    })

    # 加载数据集
    dataset = LeRobotDataset(
        repo_id="myusername/aloha-pick-place",
        root="./data/lerobot",
        delta_timestamps=delta_timestamps
    )

    # 创建优化器和数据加载器
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 训练循环
    step = 0
    done = False
    print(f"Starting training for {training_steps} steps...")

    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps}, Loss: {loss.item():.4f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # 保存模型
    print(f"Saving model to {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("Training complete!")

if __name__ == "__main__":
    main()
```

运行脚本：

```bash
python train_smolvla.py
```

### 从预训练模型微调

```bash
lerobot-train \
    --policy-name smolvla \
    --pretrained-model-path lerobot/smolvla_base \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --output-dir outputs/smolvla_aloha_finetuned \
    --num-train-iters 5000 \
    --batch-size 8
```

### 评估训练好的模型

```bash
lerobot-eval \
    --policy-path outputs/smolvla_aloha \
    --env aloha \
    --num-episodes 10 \
    --output-dir outputs/eval_results
```

### 使用训练好的模型进行推理

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# 加载模型
model_path = "outputs/smolvla_aloha"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(model_path)
policy.eval()
policy.to(device)

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_path,
    preprocessor_overrides={"device_processor": {"device": str(device)}}
)

# 推理示例（需要连接真实机器人或仿真环境）
# observation = get_observation_from_robot()
# obs_frame = build_inference_frame(observation, ...)
# obs = preprocess(obs_frame)
# action = policy.select_action(obs)
# action = postprocess(action)
# send_action_to_robot(action)
```

---

## 常见问题

### Q1: 转换时出现"No module named 'cv2'"错误

**A**: 安装OpenCV：

```bash
pip install opencv-python-headless
```

或者安装完整的lerobot依赖：

```bash
pip install -e ".[all]"
```

### Q2: 内存不足错误

**A**: 尝试减小批量大小：

```bash
lerobot-train \
    ... \
    --batch-size 4
```

或者使用梯度累积：

```bash
lerobot-train \
    ... \
    --batch-size 4 \
    --gradient-accumulation-steps 2
```

### Q3: 如何查看可用的摄像头？

**A**: 在转换后查看数据集信息：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(
    repo_id="myusername/aloha-pick-place",
    root="./data/lerobot"
)

print("Available cameras:", meta.video_keys)
```

### Q4: 训练时GPU利用率低

**A**:
1. 增加数据加载的worker数量：`--num-workers 8`
2. 使用更大的批量大小
3. 启用混合精度训练（如果使用accelerate）

### Q5: 如何恢复中断的训练？

**A**: 使用checkpoint恢复：

```bash
lerobot-train \
    --resume-from outputs/smolvla_aloha/checkpoint-5000 \
    ...其他参数...
```

### Q6: 数据集转换需要多长时间？

**A**: 取决于：
- Episode数量和长度
- 图像分辨率
- 硬盘速度

参考：50个episodes（每个~200帧，2个摄像头640x480）大约需要5-15分钟。

### Q7: 转换后的数据集大小是多少？

**A**:
- 使用视频压缩（MP4）：约为原HDF5的30-50%
- 不使用视频：约为原HDF5的10-20%（仅parquet）

### Q8: 如何验证数据集格式正确？

**A**: 运行验证：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset(
        repo_id="myusername/aloha-pick-place",
        root="./data/lerobot"
    )
    print("✓ Dataset loaded successfully!")
    print(f"  Total episodes: {dataset.meta.total_episodes}")
    print(f"  Total frames: {dataset.meta.total_frames}")

    # 测试读取一帧
    sample = dataset[0]
    print("✓ Successfully read first frame!")

except Exception as e:
    print(f"✗ Error loading dataset: {e}")
```

---

## 额外资源

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLA Model Card](https://huggingface.co/lerobot/smolvla_base)
- [LeRobot Discord](https://discord.gg/s3KuuzsPFb)

---

## 完整工作流示例

以下是一个完整的端到端工作流：

```bash
# 1. 创建环境
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 2. 安装LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 3. 转换数据集
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha_recordings \
    --instruction-dir ./data/instructions \
    --repo-id myusername/aloha-pick-place \
    --output-dir ./data/lerobot/aloha-pick-place

# 4. 验证数据集
lerobot-dataset-viz \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --mode local \
    --episode-index 0

# 5. 训练SmolVLA
lerobot-train \
    --policy-name smolvla \
    --repo-id myusername/aloha-pick-place \
    --root ./data/lerobot \
    --output-dir outputs/smolvla_aloha \
    --num-train-iters 10000 \
    --batch-size 8 \
    --eval-freq 1000 \
    --save-freq 1000 \
    --use-wandb \
    --wandb-project my-robotics-project

# 6. 评估模型
lerobot-eval \
    --policy-path outputs/smolvla_aloha \
    --env aloha \
    --num-episodes 10
```

祝你训练顺利！🚀
