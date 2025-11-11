# ALOHA HDF5 to LeRobot v3.0 Converter

This script converts ALOHA datasets stored in HDF5 format to the LeRobot v3.0 dataset format.

## Prerequisites

- ALOHA HDF5 dataset files (*.hdf5)
- Instruction JSON files for each episode
- Python 3.10+
- LeRobot installed

## Instruction File Format

The instruction directory should contain JSON files named `episode{N}.json` with the following format:

```json
{
  "seen": [
    "instruction 1",
    "instruction 2",
    "instruction 3"
  ]
}
```

The script will randomly select one instruction from the "seen" list for each episode.

## Usage

### Basic Usage

Convert all episodes in a directory:

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir /path/to/hdf5/files \
    --instruction-dir /path/to/instructions \
    --repo-id username/dataset-name \
    --output-dir /path/to/output
```

### Usage Examples

#### 1. Convert to Custom Output Directory

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha \
    --instruction-dir ./data/instructions \
    --repo-id myuser/aloha-dataset \
    --output-dir ./data/lerobot/aloha-dataset
```

#### 2. Convert Specific Episodes

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha \
    --instruction-dir ./data/instructions \
    --repo-id myuser/aloha-dataset \
    --output-dir ./data/lerobot/aloha-dataset \
    --episodes 0 1 2 3 4
```

#### 3. Convert and Push to Hub

```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --raw-dir ./data/raw/aloha \
    --instruction-dir ./data/instructions \
    --repo-id myuser/aloha-dataset \
    --push-to-hub
```

Note: When pushing to hub without specifying `--output-dir`, the dataset will be saved to `~/.cache/huggingface/lerobot/{repo-id}` by default.

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--raw-dir` | Yes | Directory containing HDF5 episode files |
| `--instruction-dir` | Yes | Directory containing instruction JSON files |
| `--repo-id` | Yes | Repository identifier (e.g., 'username/dataset-name') |
| `--output-dir` | No | Output directory for the converted dataset (default: `~/.cache/huggingface/lerobot/{repo-id}`) |
| `--episodes` | No | List of specific episode indices to convert (default: all episodes) |
| `--push-to-hub` | No | Flag to upload the converted dataset to Hugging Face Hub |

## HDF5 File Structure

The script expects HDF5 files with the following structure:

```
/observation/
    {camera_name}/
        rgb/                    # Array of images (compressed or uncompressed)
/joint_action/
    vector/                     # Array of joint states and actions [shape: (num_frames, 14)]
```

### Expected Data Format

- **Joint Actions**: 14-dimensional vectors containing:
  - Left arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper (7 dims)
  - Right arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper (7 dims)

- **Camera Images**: RGB images in format:
  - Compressed: JPEG/PNG encoded bytes that can be decoded with OpenCV
  - Uncompressed: 4D numpy array (num_frames, height, width, channels)

## Output Dataset Structure

The converted dataset will have the following structure:

```
output_dir/
├── data/
│   └── chunk-000/
│       └── file-000.parquet          # Contains states and actions
├── videos/
│   └── {camera_name}/
│       └── chunk-000/
│           └── file-000.mp4          # Video data for each camera
└── meta/
    ├── info.json                      # Dataset metadata
    ├── episodes/                      # Episode metadata (parquet)
    ├── tasks/                         # Task information (parquet)
    └── episodes_stats/                # Per-episode statistics (parquet)
```

## Features

- ✅ Automatic camera detection from HDF5 files
- ✅ Support for both compressed and uncompressed images
- ✅ Automatic image resizing to 640x480
- ✅ Channel-first format conversion (CHW) for PyTorch compatibility
- ✅ Episode-level instruction support
- ✅ Video encoding for efficient storage
- ✅ Compatible with LeRobot v3.0 dataset API
- ✅ Progress tracking with detailed logging

## Verification

After conversion, verify the dataset with:

### 1. Using Visualization Tool

```bash
lerobot-dataset-viz \
    --repo-id username/dataset-name \
    --root /path/to/output \
    --mode local \
    --episode-index 0
```

### 2. Using Python

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load the converted dataset
dataset = LeRobotDataset(
    repo_id="username/dataset-name",
    root="/path/to/output"
)

print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Total frames: {dataset.meta.total_frames}")
print(f"FPS: {dataset.meta.info['fps']}")
print(f"Robot type: {dataset.meta.info['robot_type']}")
print(f"Features: {list(dataset.features.keys())}")

# Test reading a frame
sample = dataset[0]
for key, value in sample.items():
    if hasattr(value, 'shape'):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")
```

## Technical Details

### Dataset Features Schema

The script creates the following features:

```python
{
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),  # 14 joint positions (7 per arm)
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
        "names": { ... }  # Same as observation.state
    },
    "observation.images.{camera_name}": {
        "dtype": "video",
        "shape": (3, 480, 640),  # CHW format
        "names": ["channels", "height", "width"]
    }
}
```

### Dataset Metadata

The script sets:
- **FPS**: 30 (ALOHA standard)
- **Robot Type**: "aloha"
- **Codebase Version**: v3.0

### API Compatibility

This script uses the official LeRobot v3.0 API:

```python
# Create dataset
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    robot_type="aloha",
    fps=30,
    features=features,
    root=output_dir  # Custom output directory support
)

# Add frames
dataset.add_frame(frame_dict)

# Save episode
dataset.save_episode()

# Finalize (closes parquet writers)
dataset.finalize()

# Optional: push to hub
dataset.push_to_hub()
```

## Troubleshooting

### Issue: "No module named 'cv2'"

**Solution**: Install OpenCV:
```bash
pip install opencv-python-headless
```

### Issue: "No module named 'h5py'"

**Solution**: Install h5py:
```bash
pip install h5py
```

### Issue: Memory error during conversion

**Solution**: Convert episodes in batches:
```bash
python examples/port_datasets/port_aloha_hdf5.py \
    --episodes 0 1 2 3 4 \
    ...

python examples/port_datasets/port_aloha_hdf5.py \
    --episodes 5 6 7 8 9 \
    ...
```

### Issue: "FileNotFoundError: Instruction file not found"

**Solution**: Ensure instruction JSON files are named correctly:
- `episode0.json` for `episode0.hdf5`
- `episode1.json` for `episode1.hdf5`
- etc.

### Issue: Images have wrong dimensions

The script automatically resizes images to 640x480. If you need different dimensions, modify the `get_aloha_features()` function in the script.

## Performance

Conversion speed depends on:
- Number of episodes and frames
- Image resolution
- CPU/GPU availability
- Disk I/O speed

**Approximate timing**:
- 50 episodes × 200 frames × 2 cameras (640×480): ~5-15 minutes on modern hardware

## Next Steps

After converting your dataset:

1. **Verify the conversion**: Use the visualization tool or load the dataset in Python
2. **Train a policy**: See [ALOHA_SMOLVLA_GUIDE.md](../../ALOHA_SMOLVLA_GUIDE.md) for training instructions
3. **Share your dataset**: Use `--push-to-hub` to share with the community

## Related Documentation

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Training Guide](../../ALOHA_SMOLVLA_GUIDE.md)
- [Dataset Format Specification](https://huggingface.co/docs/lerobot/dataset_format)

## Support

For issues or questions:
- GitHub Issues: https://github.com/huggingface/lerobot/issues
- Discord: https://discord.gg/s3KuuzsPFb
