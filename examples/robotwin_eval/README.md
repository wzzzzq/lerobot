# RoboTwin Evaluation Scripts for SmolVLA

This directory contains simplified evaluation scripts for running SmolVLA policies on RoboTwin simulation tasks.

## Overview

The evaluation scripts use the main LeRobot SmolVLA implementation (from `src/lerobot/policies/smolvla/`) instead of a nested copy, making the codebase cleaner and easier to maintain.

## Directory Structure

```
examples/robotwin_eval/
├── eval_policy_smolvla.py       # Main evaluation script
├── eval_put_bottles_dustbin.sh  # Bash script for put_bottles_dustbin task
├── eval_stack_blocks.sh         # Bash script for stack_blocks_three task
└── README.md                    # This file
```

## Requirements

1. **RoboTwin Environment**: The RoboTwin directory must be present in the repository root
2. **SAPIEN Simulator**: Required for running RoboTwin environments
3. **Trained SmolVLA Policy**: A checkpoint directory with `pretrained_model/` subdirectory

## Quick Start

### 1. Using Bash Scripts (Recommended)

Edit the configuration variables at the top of the bash script:

```bash
# Edit policy path and settings
vim examples/robotwin_eval/eval_put_bottles_dustbin.sh

# Run evaluation
bash examples/robotwin_eval/eval_put_bottles_dustbin.sh
```

### 2. Using Python Directly

Create a configuration YAML file:

```yaml
# config.yml
policy_path: /path/to/checkpoints
ckpt_setting: "last"
device: cuda
task_name: put_bottles_dustbin
task_config: demo_clean
instruction_type: seen
num_episodes: 10
seed: 42
n_action_steps: 10
policy_name: SmolVLA
tokenizer_max_length: 96
```

Run evaluation:

```bash
cd RoboTwin
python ../examples/robotwin_eval/eval_policy_smolvla.py --config config.yml
```

## Configuration Options

### Policy Settings

- `policy_path`: Base path to policy checkpoints (e.g., `/path/to/training/checkpoints`)
- `ckpt_setting`: Which checkpoint to load
  - `"last"`: Load from `policy_path/last/pretrained_model/`
  - `"best"`: Load from `policy_path/best/pretrained_model/`
  - Custom path: Direct path to checkpoint directory
- `device`: Device to run on (`"cuda"` or `"cpu"`)

### Task Settings

- `task_name`: RoboTwin task name
  - `put_bottles_dustbin`
  - `stack_blocks_three`
  - `stack_bowls_three`
  - `place_dual_shoes`
  - `place_empty_cup`
- `task_config`: Task configuration variant
  - `demo_clean`: Clean environment
  - `randomized`: Randomized environment with domain randomization
- `instruction_type`: Type of language instructions
  - `seen`: Instructions seen during training
  - `unseen`: Novel instructions for generalization testing

### Evaluation Settings

- `num_episodes`: Number of episodes to evaluate (default: 10)
- `seed`: Random seed for reproducibility
- `n_action_steps`: Number of action steps to execute per policy query (default: 10)

## Output

Evaluation results are saved to:

```
eval_result/
└── <task_name>/
    └── SmolVLA/
        └── <task_config>/
            └── <ckpt_setting>/
                └── <timestamp>/
                    ├── _result.txt        # Success rate and scores
                    └── episode*.mp4       # Videos (if enabled)
```

Example `_result.txt`:

```
Timestamp: 2025-01-15_14-30-00

Instruction Type: seen

Success Rate: 8/10 = 80.0%
Task Score: 0.875
```

## Available RoboTwin Tasks

| Task Name | Description |
|-----------|-------------|
| `put_bottles_dustbin` | Place bottles into a dustbin |
| `stack_blocks_three` | Stack three blocks |
| `stack_bowls_three` | Stack three bowls |
| `place_dual_shoes` | Place a pair of shoes |
| `place_empty_cup` | Place an empty cup |

## Advanced Usage

### Command-Line Overrides

You can override config values from the command line:

```bash
python eval_policy_smolvla.py \
    --config config.yml \
    --overrides --num_episodes 20 --seed 123
```

### Custom Environment Variables

Some configurations may require setting environment variables:

```bash
# Temporary directory for SAPIEN
export TMPDIR=/path/to/tmp

# Proxy settings (if needed)
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port

# GPU selection
export CUDA_VISIBLE_DEVICES=0
```

### Video Recording

To enable video recording, ensure `eval_video_log: true` is set in the task config file (e.g., `RoboTwin/task_config/demo_clean.yml`).

## Key Differences from Original RoboTwin Eval

1. **No Nested LeRobot**: Uses main repo's SmolVLA instead of bundled copy
2. **Simplified Interface**: Single wrapper class handles policy adaptation
3. **Cleaner Imports**: Direct imports from `lerobot.policies.smolvla`
4. **Better Logging**: Colored output and progress tracking
5. **Bash Scripts**: Convenient pre-configured evaluation scripts

## Troubleshooting

### Import Errors

If you see import errors related to RoboTwin:

```python
ModuleNotFoundError: No module named 'envs'
```

Make sure you're running the script from the correct directory (see Quick Start).

### Missing Checkpoint

```
FileNotFoundError: Policy checkpoint not found: /path/to/checkpoint
```

Check that your checkpoint directory has this structure:

```
checkpoints/
├── last/
│   └── pretrained_model/
│       ├── config.json
│       ├── model.safetensors
│       └── ... (other model files)
└── best/
    └── pretrained_model/
        └── ... (same structure)
```

### SAPIEN Errors

If SAPIEN fails to initialize:

1. Check that SAPIEN is properly installed
2. Set `TMPDIR` to a writable location
3. Ensure you have proper GPU access

## Example Workflows

### Evaluate Multiple Seeds

```bash
for seed in 42 123 456; do
    python eval_policy_smolvla.py --config config.yml \
        --overrides --seed $seed
done
```

### Evaluate on All Task Configs

```bash
for config in demo_clean randomized; do
    python eval_policy_smolvla.py --config config.yml \
        --overrides --task_config $config
done
```

### Compare Checkpoints

```bash
for ckpt in last best; do
    python eval_policy_smolvla.py --config config.yml \
        --overrides --ckpt_setting $ckpt
done
```

## Contributing

When adding new tasks:

1. Create a new bash script (e.g., `eval_new_task.sh`)
2. Update task-specific settings (task_name, default policy path)
3. Test the evaluation
4. Update this README with task information

## References

- [RoboTwin Paper](https://arxiv.org/abs/2409.02920)
- [SmolVLA Documentation](../../docs/source/policy_smolvla_README.md)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
