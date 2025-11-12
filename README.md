# LeRobot Fork - ALOHA 数据转换和 SmolVLA 训练工具

本仓库是 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 的 Fork，专注于 ALOHA 机器人数据集的转换和 SmolVLA 模型训练。

## 🎯 主要功能

- ✅ **ALOHA HDF5 转换器**：将 ALOHA HDF5 数据集转换为 LeRobot v3.0 格式
- ✅ **SmolVLA 训练支持**：完整的 SmolVLA 模型训练流程和指南
- ✅ **自定义输出目录**：灵活的数据集输出路径配置
- ✅ **完整中文文档**：详细的使用指南和故障排除

## 📦 快速开始

### 环境设置

```bash
# 1. 创建虚拟环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 2. 安装 FFmpeg
conda install ffmpeg -c conda-forge

# 3. 克隆仓库
git clone https://github.com/wzzzzq/lerobot.git
cd lerobot

# 4. 安装 LeRobot
pip install -e .

# 5. 安装 SmolVLA 依赖
pip install -e ".[smolvla]"
```

### 数据转换

将 ALOHA HDF5 数据集转换为 LeRobot 格式：

```bash
# 转换 RoboTwin stack_blocks_two 数据集（本地保存，无需 push to hub）
export TMPDIR=/pfs/pfs-ilWc5D/ziqianwang/tmp
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --instruction-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --repo-id robotwin/stack_blocks_two \
    --output-dir /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --resume

# 通用格式（替换路径）
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /path/to/hdf5/files \
    --instruction-dir /path/to/instructions \
    --repo-id myusername/aloha-dataset \
    --output-dir /path/to/output
```

### 训练 SmolVLA

使用转换后的数据集训练 SmolVLA 模型：

```bash
# 使用提供的训练脚本（推荐）
bash examples/train_smolvla.sh

# 或者直接使用命令行
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=robotwin/stack_blocks_two \
    --dataset.root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --output_dir=/pfs/pfs-ilWc5D/ziqianwang/pretrain_stack_blocks_two \
    --steps=10000 \
    --batch_size=8
```

**提示**：
- RTX 4000 系列 GPU 必须设置 `NCCL_P2P_DISABLE="1"` 和 `NCCL_IB_DISABLE="1"`
- 可以编辑 `examples/train_smolvla.sh` 来自定义训练参数
- 取消脚本中的注释来启用 W&B 日志记录

## 📚 详细文档

### 1. ALOHA HDF5 数据转换

#### 准备数据

**HDF5 文件结构**：

支持两种格式：

1. **ALOHA 原始格式**：
```
/observation/
    {camera_name}/
        rgb/                    # 图像数组（压缩或未压缩）
/joint_action/
    vector/                     # 关节状态和动作 [14维]
```

2. **RoboTwin 格式**：
```
/observations/
    images/
        {camera_name}/          # 压缩图像数据
    qpos/                       # 关节状态 [14维]
/action/                        # 动作 [14维]
```

**指令文件格式**：

支持两种格式：

1. **标准格式**（`episode0.json`, `episode1.json` 等）：
```json
{
  "seen": [
    "pick up the red block",
    "grasp the red object"
  ]
}
```

2. **RoboTwin 格式**（`episode_0/instructions.json`, `episode_1/instructions.json` 等）：
```json
{
  "instructions": [
    "Move red block and green block to the center",
    "Stack green block above red block"
  ]
}
```

#### 转换命令

**基本转换（本地保存）**：
```bash
# RoboTwin stack_blocks_two 数据集示例
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --instruction-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --repo-id robotwin/stack_blocks_two \
    --output-dir /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two

# 通用格式
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha \
    --instruction-dir ./data/instructions \
    --repo-id myuser/aloha-dataset \
    --output-dir ./data/lerobot/aloha-dataset
```

**转换特定 episodes**：
```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --instruction-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --repo-id robotwin/stack_blocks_two \
    --output-dir /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --episodes 0 1 2 3 4
```

**转换并上传到 Hub**（如需要）：
```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha \
    --instruction-dir ./data/instructions \
    --repo-id myuser/aloha-dataset \
    --push-to-hub
```

#### 命令行参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--raw-dir` | 是 | HDF5 文件目录 |
| `--instruction-dir` | 是 | 指令 JSON 文件目录 |
| `--repo-id` | 是 | 数据集标识符（如 `username/dataset-name`） |
| `--output-dir` | 否 | 输出目录（默认：`~/.cache/huggingface/lerobot/{repo-id}`） |
| `--episodes` | 否 | 要转换的 episode 索引列表（默认：全部） |
| `--push-to-hub` | 否 | 上传到 Hugging Face Hub |

#### 验证转换结果

**使用可视化工具**：
```bash
lerobot-dataset-viz \
    --repo-id robotwin/stack_blocks_two \
    --root /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --mode local \
    --episode-index 0
```

**使用 Python**：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="robotwin/stack_blocks_two",
    root="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two"
)

print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Total frames: {dataset.meta.total_frames}")
print(f"FPS: {dataset.meta.info['fps']}")
print(f"Features: {list(dataset.features.keys())}")
```

#### 输出数据集结构

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
    ├── episodes/                      # Episode 元数据
    ├── tasks/                         # 任务信息
    └── episodes_stats/                # Episode 统计信息
```

### 2. SmolVLA 训练

#### 方法 1：使用训练脚本（推荐）

**使用提供的训练脚本**：
```bash
# 编辑 examples/train_smolvla.sh 来配置参数
# 然后运行：
bash examples/train_smolvla.sh
```

**脚本配置说明**：
- `DATASET_REPO_ID`: 数据集标识符
- `DATASET_ROOT`: 数据集本地路径
- `OUTPUT_DIR`: 模型输出目录
- `CUDA_DEVICE`: 使用的 GPU 设备编号
- `BATCH_SIZE`: 批量大小
- `STEPS`: 训练步数

取消注释 W&B 相关行来启用训练日志记录。

#### 方法 2：使用命令行

**基本训练**：
```bash
# 使用 stack_blocks_two 数据集
# 注意：RTX 4000 系列 GPU 需要设置 NCCL 环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=robotwin/stack_blocks_two \
    --dataset.root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --output_dir=outputs/smolvla_stack_blocks \
    --steps=10000 \
    --batch_size=8 \
    --eval_freq=1000 \
    --save_freq=1000 \
    --log_freq=100
```

**使用 W&B 跟踪**：
```bash
# 注意：RTX 4000 系列 GPU 需要设置 NCCL 环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

wandb login

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=robotwin/stack_blocks_two \
    --dataset.root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --output_dir=outputs/smolvla_stack_blocks \
    --steps=10000 \
    --batch_size=8 \
    --wandb.enable=true \
    --wandb.project=robotwin-training \
    --wandb.run_id=smolvla-stack-blocks-run1
```

**多 GPU 训练**：
```bash
# 注意：RTX 4000 系列 GPU 需要设置 NCCL 环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

accelerate launch --multi_gpu --num_processes=4 \
    src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --dataset.repo_id=robotwin/stack_blocks_two \
    --dataset.root=/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --output_dir=outputs/smolvla_stack_blocks \
    --steps=10000 \
    --batch_size=32
```

**从预训练模型微调**：
```bash
# 注意：RTX 4000 系列 GPU 需要设置 NCCL 环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=myuser/aloha-dataset \
    --dataset.root=/path/to/output \
    --output_dir=outputs/smolvla_aloha_finetuned \
    --steps=5000 \
    --batch_size=8
```

#### 方法 3：使用 Python 脚本

**从预训练模型微调**：
```bash
# 注意：RTX 4000 系列 GPU 需要设置 NCCL 环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=myuser/aloha-dataset \
    --dataset.root=/path/to/output \
    --output_dir=outputs/smolvla_aloha_finetuned \
    --steps=5000 \
    --batch_size=8
```

#### 方法 2：使用 Python 脚本

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

    # 加载数据集元数据
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="robotwin/stack_blocks_two",
        root="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two"
    )

    # 准备特征配置
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 创建策略
    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features)
    policy = SmolVLAPolicy(cfg)
    policy.train()
    policy.to(device)

    # 创建处理器
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    # 准备 delta_timestamps
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
        repo_id="robotwin/stack_blocks_two",
        root="/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two",
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

            if step % 100 == 0:
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

运行：
```bash
python train_smolvla.py
```

#### 评估模型

```bash
lerobot-eval \
    --policy-path outputs/smolvla_aloha \
    --env aloha \
    --num-episodes 10 \
    --output-dir outputs/eval_results
```

#### 推理

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

# 推理（需要连接机器人或仿真环境）
# observation = get_observation_from_robot()
# obs_frame = build_inference_frame(observation, ...)
# obs = preprocess(obs_frame)
# action = policy.select_action(obs)
# action = postprocess(action)
# send_action_to_robot(action)
```

## 🔧 技术细节

### 数据集特征 Schema

```python
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),  # 14个关节位置（每臂7个）
        "names": {
            "axes": [
                "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
                "left_wrist_angle", "left_wrist_rotate", "left_gripper",
                "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
                "right_wrist_angle", "right_wrist_rotate", "right_gripper"
            ]
        }
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": { ... }  # 与 observation.state 相同
    },
    "observation.images.{camera_name}": {
        "dtype": "video",
        "shape": (3, 480, 640),  # CHW 格式
        "names": ["channels", "height", "width"]
    }
}
```

### API 兼容性

本工具使用 LeRobot v3.0 官方 API：

```python
# 创建数据集
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    robot_type="aloha",
    fps=30,
    features=features,
    root=output_dir  # 支持自定义输出目录
)

# 添加帧
dataset.add_frame(frame_dict)

# 保存 episode
dataset.save_episode()

# 完成（关闭写入器）
dataset.finalize()

# 可选：推送到 Hub
dataset.push_to_hub()
```

## ❓ 常见问题

### Q1: 转换时出现 "No module named 'cv2'" 错误

**A**: 安装 OpenCV：
```bash
pip install opencv-python-headless
```

### Q2: 内存不足错误

**A**: 减小批量大小：
```bash
python src/lerobot/scripts/lerobot_train.py --batch_size=4 ...
```

或使用梯度累积：
```bash
# LeRobot 使用 gradient_accumulation_steps 需要通过配置文件设置
python src/lerobot/scripts/lerobot_train.py --batch_size=4 ...
```

### Q3: 如何查看可用的摄像头？

**A**: 转换后查看数据集信息：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(
    repo_id="myuser/aloha-dataset",
    root="./data/lerobot"
)
print("Available cameras:", meta.video_keys)
```

### Q4: 训练时 GPU 利用率低

**A**:
1. 增加数据加载的 worker 数量
2. 使用更大的批量大小
3. 启用混合精度训练

### Q5: 如何恢复中断的训练？

**A**: 使用 checkpoint 恢复：
```bash
python src/lerobot/scripts/lerobot_train.py \
    --resume=true \
    --checkpoint_path=outputs/smolvla_aloha/checkpoint-5000 \
    ...其他参数...
```

### Q6: 数据集转换需要多长时间？

**A**: 参考（50个episodes，每个~200帧，2个摄像头640x480）：约5-15分钟

### Q7: 转换后的数据集大小是多少？

**A**:
- 使用视频压缩（MP4）：约为原 HDF5 的 30-50%
- 不使用视频：约为原 HDF5 的 10-20%

### Q8: 如何验证数据集格式正确？

**A**: 运行验证：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    dataset = LeRobotDataset(
        repo_id="myuser/aloha-dataset",
        root="./data/lerobot"
    )
    print("✓ Dataset loaded successfully!")
    sample = dataset[0]
    print("✓ Successfully read first frame!")
except Exception as e:
    print(f"✗ Error: {e}")
```

## 🚀 完整工作流示例

```bash
# 1. 环境设置
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 2. 安装
git clone https://github.com/wzzzzq/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 3. 转换 RoboTwin stack_blocks_two 数据集
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --instruction-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/stack_blocks_two-demo_clean-700 \
    --repo-id robotwin/stack_blocks_two \
    --output-dir /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two

# 4. 验证数据集
lerobot-dataset-viz \
    --repo-id robotwin/stack_blocks_two \
    --root /pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/stack_blocks_two \
    --mode local \
    --episode-index 0

# 5. 训练 SmolVLA（使用提供的脚本）
# 编辑 examples/train_smolvla.sh 配置参数，然后运行：
bash examples/train_smolvla.sh

# 6. 评估模型
lerobot-eval \
    --policy-path outputs/smolvla_stack_blocks \
    --env aloha \
    --num-episodes 10
```

## 📊 性能基准

| 配置 | Episodes | 帧数 | 摄像头 | 转换时间 | 数据集大小 |
|------|----------|------|--------|----------|------------|
| 小型 | 50 | ~200/ep | 2个 | ~5-10分钟 | ~2-3 GB |
| 中型 | 200 | ~200/ep | 2个 | ~20-40分钟 | ~8-12 GB |
| 大型 | 500 | ~300/ep | 3个 | ~60-120分钟 | ~30-50 GB |

## 🤝 贡献

本仓库基于 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) v0.4.1

### 主要改进

- ✅ ALOHA HDF5 到 LeRobot v3.0 转换器
- ✅ 自定义输出目录支持
- ✅ 完整的中文文档
- ✅ SmolVLA 训练指南
- ✅ 故障排除和最佳实践

## 📄 许可证

Apache 2.0 License - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关资源

- [LeRobot 官方文档](https://huggingface.co/docs/lerobot)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLA Model Card](https://huggingface.co/lerobot/smolvla_base)
- [Hugging Face Hub](https://huggingface.co/lerobot)

## 📧 支持

遇到问题？
- GitHub Issues: https://github.com/wzzzzq/lerobot/issues
- LeRobot Discord: https://discord.gg/s3KuuzsPFb

---

⭐ 如果这个项目对你有帮助，请给个 Star！
