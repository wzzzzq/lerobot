#!/usr/bin/env python3
"""
Simplified SmolVLA policy evaluation script for RoboTwin environments.

This script uses the main lerobot SmolVLA implementation instead of a nested copy.
It provides a minimal interface to evaluate SmolVLA policies in RoboTwin tasks.

Usage:
    python examples/robotwin_eval/eval_policy_smolvla.py --config config.yml
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any

# Add RoboTwin to path
LEROBOT_ROOT = Path(__file__).parent.parent.parent
ROBOTWIN_ROOT = LEROBOT_ROOT / "RoboTwin"
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT / "description" / "utils"))

# Import from main lerobot (not nested copy)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import RoboTwin components
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
import importlib
import traceback
import subprocess


class SmolVLAWrapper:
    """Wrapper to adapt main lerobot SmolVLA for RoboTwin interface."""

    def __init__(self, policy: SmolVLAPolicy, preprocessor, postprocessor, device: str = "cuda"):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.observation_window = None
        self.instruction = None

    def set_language(self, instruction: str):
        """Set the language instruction."""
        self.instruction = instruction

    def update_observation_window(self, img_arr: list, state: np.ndarray):
        """
        Build observation window from images and state.

        Args:
            img_arr: List of images [head_camera, left_camera, right_camera] (matching training data order)
            state: Robot joint state vector
        """
        def prepare_img(img):
            # Convert HWC to CHW, normalize to [0, 1]
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img)

        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))

        # Build observation dict (without batch dimension - preprocessor will add it)
        observation = {
            "observation.state": state_tensor,
            "task": self.instruction if isinstance(self.instruction, str) else self.instruction[0],
        }

        # Camera names - MUST match training data order: head, left, right (NO front_camera)
        camera_names = ["head_camera", "left_camera", "right_camera"]

        # Add each camera to observation
        for i, camera_name in enumerate(camera_names):
            if i < len(img_arr):
                key = f"observation.images.{camera_name}"
                observation[key] = prepare_img(img_arr[i])

        # Use the policy's preprocessor to process the observation
        # This will add batch dimension, tokenize language, and normalize
        self.observation_window = self.preprocessor(observation)

    def get_action(self) -> np.ndarray:
        """Get action and apply postprocessor for denormalization."""
        if self.observation_window is None:
            raise ValueError("Must call update_observation_window() first!")

        # Get normalized action from policy (returns Tensor)
        action_tensor = self.policy.select_action(self.observation_window)

        # Apply postprocessor to denormalize action (Tensor → Tensor)
        # Note: PolicyAction is just a type alias for torch.Tensor, not a class
        action_denormalized = self.postprocessor(action_tensor)

        # Extract action numpy array
        action_numpy = action_denormalized.cpu().numpy().squeeze(0)

        return action_numpy

    def reset(self):
        """Reset internal state."""
        self.observation_window = None
        self.instruction = None
        self.policy.reset()  # Reset policy's internal queues


def load_model(usr_args: Dict) -> SmolVLAWrapper:
    """
    Load SmolVLA model from checkpoint.

    Args:
        usr_args: Configuration dictionary containing:
            - policy_path: Base path to policy checkpoints
            - ckpt_setting: Checkpoint to load ("last" or "best")
            - device: Device to run on ("cuda" or "cpu")
            - num_steps: (optional) Override denoising steps for inference

    Returns:
        SmolVLAWrapper instance
    """
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SmolVLA model on device: {device}")

    # Build full policy path: policy_path/ckpt_setting/pretrained_model
    policy_base_path = usr_args["policy_path"]
    ckpt_setting = usr_args.get("ckpt_setting", "last")
    policy_path = os.path.join(policy_base_path, ckpt_setting, "pretrained_model")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    # Override num_steps in config.json if specified (permanent change)
    if "num_steps" in usr_args:
        import json
        config_path = os.path.join(policy_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        num_steps = usr_args["num_steps"]
        print(f"Overriding num_steps in config.json: {config_dict.get('num_steps', 10)} -> {num_steps}")
        config_dict['num_steps'] = num_steps
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    print(f"Loading from checkpoint: {policy_path}")

    # Override num_steps in config.json if specified (permanent change)
    if "num_steps" in usr_args:
        import json
        config_path = os.path.join(policy_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        num_steps = usr_args["num_steps"]
        print(f"Overriding num_steps in config.json: {config_dict.get('num_steps', 10)} -> {num_steps}")
        config_dict['num_steps'] = num_steps

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    # Load policy (new checkpoints don't have reflow artifacts)
    policy = SmolVLAPolicy.from_pretrained(policy_path)

    policy.to(device)
    policy.eval()

    print(f"✓ Successfully loaded model from: {policy_path}")
    print(f"✓ Using num_steps: {policy.config.num_steps}")

    # Create preprocessor and postprocessor from the pretrained checkpoint
    # This will load the saved normalization statistics
    print("Loading preprocessor and postprocessor from checkpoint...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
    )
    print("✓ Successfully loaded preprocessor and postprocessor")

    # Wrap policy for RoboTwin interface
    model = SmolVLAWrapper(policy, preprocessor, postprocessor, device=device)
    return model


def encode_obs(observation: Dict) -> tuple:
    """Extract images and state from RoboTwin observation.
    
    Returns images in the order: [head_camera, left_camera, right_camera]
    This MUST match the training data camera order.
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def eval_step(TASK_ENV: Any, model: SmolVLAWrapper, observation: Dict) -> None:
    """
    Execute one evaluation step.

    Args:
        TASK_ENV: RoboTwin task environment
        model: SmolVLA model wrapper
        observation: Current observation from environment
    """
    # Set instruction on first step
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # Extract images and state from observation
    input_rgb_arr, input_state = encode_obs(observation)

    # Update observation window and get action
    model.update_observation_window(input_rgb_arr, input_state)
    action = model.get_action()

    # Execute action in environment
    TASK_ENV.take_action(action)


def reset_model(model: SmolVLAWrapper) -> None:
    """Reset model internal state."""
    if model:
        print("Resetting SmolVLA internal state...")
        model.reset()


def get_camera_config(camera_type: str) -> Dict:
    """Load camera configuration from RoboTwin config."""
    camera_config_path = ROBOTWIN_ROOT / "task_config" / "_camera_config.yml"

    if not camera_config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {camera_config_path}")

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if camera_type not in args:
        raise KeyError(f"Camera {camera_type} is not defined in config")

    return args[camera_type]


def get_embodiment_config(robot_file: str) -> Dict:
    """Load robot embodiment configuration."""
    robot_config_file = os.path.join(robot_file, "config.yml")

    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    return embodiment_args


def class_decorator(task_name: str):
    """Dynamically import and instantiate task environment."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit(f"Task {task_name} not found")
    return env_instance


def eval_policy(
    task_name: str,
    TASK_ENV: Any,
    args: Dict,
    model: SmolVLAWrapper,
    st_seed: int,
    test_num: int = 100,
    video_size: str = None,
    instruction_type: str = "seen"
) -> tuple:
    """
    Evaluate policy on task environment.

    Args:
        task_name: Name of the task
        TASK_ENV: Task environment instance
        args: Configuration dictionary
        model: SmolVLA model wrapper
        st_seed: Starting seed for evaluation
        test_num: Number of test episodes
        video_size: Video resolution for recording
        instruction_type: Type of instructions to use ("seen" or "unseen")

    Returns:
        (final_seed, success_count, average_task_score)
    """
    print(f"\033[34mTask Name: {task_name}\033[0m")
    print(f"\033[34mPolicy Name: SmolVLA\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args.get("clear_cache_freq", 10)

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args.get("render_freq", 0)
        args["render_freq"] = 0

        # Validate environment setup with expert policy
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(f" ------------- UnStableError: {e} -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(f" ------------- Error: {e}\n{stack_trace} -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        # Check if expert succeeded
        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # Setup environment for policy evaluation
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        # Setup video recording if needed
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pixel_format", "rgb24",
                    "-video_size", video_size,
                    "-framerate", "10",
                    "-i", "-",
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264",
                    "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        # Run policy evaluation
        succ = False
        reset_model(model)

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_step(TASK_ENV, model, observation)

            if TASK_ENV.eval_success:
                succ = True
                break

        task_total_reward += TASK_ENV.episode_score

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # Print results
        if succ:
            TASK_ENV.suc += 1
            print("\033[92m✓ Success!\033[0m")
        else:
            print("\033[91m✗ Fail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        # Print progress
        success_rate = TASK_ENV.suc / TASK_ENV.test_num * 100
        avg_score = TASK_ENV.episode_score / test_num
        avg_reward = task_total_reward / test_num

        print(
            f"\033[93m{task_name}\033[0m | \033[94mSmolVLA\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(success_rate, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
            f"Episode score: \033[93m{avg_score}\033[0m, Total reward: \033[92m{avg_reward}\033[0m\n"
        )

        now_seed += 1

    return now_seed, TASK_ENV.suc, task_total_reward / test_num


def main(usr_args: Dict):
    """Main evaluation function."""
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    instruction_type = usr_args.get("instruction_type", "seen")

    # Load task configuration
    task_config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    # Setup embodiment configuration
    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    # Setup camera configuration
    camera_config_path = Path(CONFIGS_PATH) / "_camera_config.yml"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # Configure dual-arm embodiment
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # Create save directory
    save_dir = Path(f"eval_result/{task_name}/SmolVLA/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup video logging
    video_save_dir = None
    video_size = None
    if args.get("eval_video_log", False):
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # Print configuration
    print("\n============= Configuration =============")
    print(f"\033[95mMessy Table:\033[0m {args['domain_randomization']['cluttered_table']}")
    print(f"\033[95mRandom Background:\033[0m {args['domain_randomization']['random_background']}")
    if args["domain_randomization"]["random_background"]:
        print(f"  - Clean Background Rate: {args['domain_randomization']['clean_background_rate']}")
    print(f"\033[95mRandom Light:\033[0m {args['domain_randomization']['random_light']}")
    if args["domain_randomization"]["random_light"]:
        print(f"  - Crazy Random Light Rate: {args['domain_randomization']['crazy_random_light_rate']}")
    print(f"\033[95mRandom Table Height:\033[0m {args['domain_randomization']['random_table_height']}")
    print(f"\033[95mRandom Head Camera Distance:\033[0m {args['domain_randomization']['random_head_camera_dis']}")
    print(f"\033[94mHead Camera Config:\033[0m {args['camera']['head_camera_type']}, {args['camera']['collect_head_camera']}")
    print(f"\033[94mWrist Camera Config:\033[0m {args['camera']['wrist_camera_type']}, {args['camera']['collect_wrist_camera']}")
    print(f"\033[94mEmbodiment Config:\033[0m {embodiment_name}")
    print("=========================================\n")

    # Load task environment
    TASK_ENV = class_decorator(task_name)

    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args.get("seed", 42)
    st_seed = 100000 * (1 + seed)
    test_num = usr_args.get("num_episodes", 100)

    # Load model and run evaluation
    model = load_model(usr_args)
    st_seed, suc_num, task_score = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type
    )

    # Save results
    file_path = save_dir / "_result.txt"
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Success Rate: {suc_num}/{test_num} = {suc_num/test_num*100:.1f}%\n")
        file.write(f"Task Score: {task_score}\n")

    print(f"\n✓ Results saved to {file_path}")
    print(f"Final success rate: {suc_num}/{test_num} = {suc_num/test_num*100:.1f}%")


def parse_args_and_config():
    """Parse command line arguments and load config."""
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA policy on RoboTwin tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Override config values")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides (format: --key value --key2 value2)
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # Test SAPIEN rendering
    from script.test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
