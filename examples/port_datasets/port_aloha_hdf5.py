#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to convert ALOHA HDF5 data to the LeRobot dataset v3.0 format.

Example usage:
    python examples/port_datasets/port_aloha_hdf5.py \
        --raw-dir /path/to/raw/data \
        --instruction-dir /path/to/instructions \
        --repo-id username/dataset-name \
        --output-dir /path/to/output

Note: The instruction directory should contain JSON files named episode{N}.json with the format:
    {"seen": ["instruction 1", "instruction 2", ...]}
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds, init_logging

# ALOHA constants
ALOHA_FPS = 30
ALOHA_ROBOT_TYPE = "aloha"


def get_aloha_features(cameras: list[str]) -> dict:
    """Define the feature schema for ALOHA dataset."""
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {
                "axes": motors,
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {
                "axes": motors,
            },
        },
    }

    # Add camera features
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    return features


def get_cameras(hdf5_file: Path) -> list[str]:
    """Extract camera names from HDF5 file."""
    with h5py.File(hdf5_file, "r") as ep:
        # Support both /observation and /observations paths
        obs_path = "/observations" if "/observations" in ep else "/observation"
        images_path = f"{obs_path}/images"
        if images_path in ep:
            return [key for key in ep[images_path].keys() if "depth" not in key]
        return [key for key in ep[obs_path].keys() if "depth" not in key]


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """Load and process images from HDF5 file for all cameras."""
    imgs_per_cam = {}
    
    # Support both /observation and /observations paths
    obs_path = "/observations" if "/observations" in ep else "/observation"
    images_path = f"{obs_path}/images"
    
    for camera in cameras:
        # Try different possible image paths
        possible_paths = [
            f"{images_path}/{camera}",
            f"{obs_path}/{camera}/rgb",
            f"{obs_path}/{camera}",
        ]
        
        img_dataset = None
        for path in possible_paths:
            if path in ep:
                img_dataset = ep[path]
                break
        
        if img_dataset is None:
            raise ValueError(f"Could not find image data for camera {camera}")
        
        # Check if images are uncompressed (4D array) or compressed (need decoding)
        uncompressed = img_dataset.ndim == 4

        if uncompressed:
            # Load all images in RAM
            imgs_array = img_dataset[:]
        else:
            # Load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in img_dataset:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_array.append(img)
            imgs_array = np.array(imgs_array)

        # Resize images to standard size if needed
        if imgs_array.shape[1:3] != (480, 640):
            imgs_array = [cv2.resize(img, (640, 480)) for img in imgs_array]
            imgs_array = np.stack(imgs_array, axis=0)

        # Convert to channel-first format (N, H, W, C) -> (N, C, H, W)
        imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
        imgs_per_cam[camera] = imgs_array

    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """Load state, action, and images from a single HDF5 episode file."""
    with h5py.File(ep_path, "r") as ep:
        # Support different data structures
        # Try /observations/qpos first (RoboTwin format), then /joint_action/vector (ALOHA format)
        if "/observations/qpos" in ep:
            state = torch.from_numpy(ep["/observations/qpos"][:]).float()
        elif "/joint_action/vector" in ep:
            state = torch.from_numpy(ep["/joint_action/vector"][:]).float()
        else:
            raise ValueError("Could not find state data in HDF5 file")
        
        # Try /action first (RoboTwin format), then /joint_action/vector (ALOHA format)
        if "/action" in ep:
            action = torch.from_numpy(ep["/action"][:]).float()
        elif "/joint_action/vector" in ep:
            action = torch.from_numpy(ep["/joint_action/vector"][:]).float()
        else:
            raise ValueError("Could not find action data in HDF5 file")

        imgs_per_cam = load_raw_images_per_camera(ep, get_cameras(ep_path))

    return imgs_per_cam, state, action


def load_instruction(instruction_dir: Path, episode_idx: int) -> str:
    """Load instruction for a given episode from JSON file."""
    json_path = instruction_dir / f"episode{episode_idx}.json"
    if not json_path.exists():
        # Try with underscore format
        json_path = instruction_dir / f"episode_{episode_idx}" / "instructions.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Instruction file not found: {json_path}")

    with open(json_path) as f_instr:
        instruction_dict = json.load(f_instr)
        # Support both "seen" and "instructions" keys
        if "instructions" in instruction_dict:
            instructions = instruction_dict["instructions"]
        elif "seen" in instruction_dict:
            instructions = instruction_dict["seen"]
        else:
            raise ValueError(f"Could not find instructions in {json_path}")
        # Randomly select one instruction from the available options
        instruction = np.random.choice(instructions)

    return instruction


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    instruction_dir: Path,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """Populate the LeRobot dataset with data from HDF5 files."""
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    start_time = time.time()
    num_episodes = len(episodes)

    for idx, ep_idx in enumerate(tqdm.tqdm(episodes, desc="Converting episodes")):
        elapsed_time = time.time() - start_time
        d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)

        logging.info(
            f"{idx + 1} / {num_episodes} episodes processed (after {d} days, {h} hours, {m} minutes, {s:.3f} seconds)"
        )

        ep_path = hdf5_files[ep_idx]
        logging.info(f"Processing episode {ep_idx}: {ep_path.name}")

        # Load episode data
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        # Load instruction
        instruction = load_instruction(instruction_dir, ep_idx)
        logging.info(f"  Instruction: {instruction}")
        logging.info(f"  Frames: {num_frames}")

        # Add all frames for this episode
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            # Add camera observations
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            dataset.add_frame(frame)

        dataset.save_episode()
        logging.info(f"  Episode {ep_idx} saved")

    return dataset


def port_aloha_hdf5(
    raw_dir: Path,
    instruction_dir: Path,
    repo_id: str,
    output_dir: Path | None = None,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
):
    """
    Port ALOHA HDF5 dataset to LeRobot v3.0 format.

    Args:
        raw_dir: Directory containing HDF5 files
        instruction_dir: Directory containing instruction JSON files
        repo_id: Repository ID for the dataset (e.g., "username/dataset-name")
        output_dir: Output directory for the converted dataset (default: ~/.cache/huggingface/lerobot/{repo_id})
        episodes: List of episode indices to convert (None = all episodes)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    init_logging()

    # Clean up existing dataset if it exists
    if output_dir is not None and output_dir.exists():
        logging.info(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    # Find all HDF5 files
    if not raw_dir.exists():
        raise ValueError(f"Raw directory does not exist: {raw_dir}")

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            file_path = Path(root) / filename
            hdf5_files.append(file_path)

    # Sort by episode number
    def extract_episode_number(path):
        # Try different patterns: episode_0.hdf5, episode0.hdf5, etc.
        match = re.search(r"episode[_]?(\d+)\.hdf5", path.name)
        if match:
            return int(match.group(1))
        return 0

    hdf5_files = sorted(hdf5_files, key=extract_episode_number)

    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir}")

    logging.info(f"Found {len(hdf5_files)} HDF5 files")

    # Get camera names from first file
    cameras = get_cameras(hdf5_files[0])
    logging.info(f"Detected cameras: {cameras}")

    # Create dataset features
    features = get_aloha_features(cameras)

    logging.info(f"Creating dataset with repo_id: {repo_id}")
    if output_dir:
        logging.info(f"Output directory: {output_dir}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=ALOHA_ROBOT_TYPE,
        fps=ALOHA_FPS,
        features=features,
        root=output_dir,
    )

    # Populate dataset
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        instruction_dir,
        episodes=episodes,
    )

    # Finalize dataset
    logging.info("Finalizing dataset...")
    dataset.finalize()
    logging.info("Dataset finalized successfully!")

    # Push to hub if requested
    if push_to_hub:
        logging.info(f"Pushing dataset to hub: {repo_id}")
        dataset.push_to_hub()
        logging.info("Dataset pushed successfully!")
    else:
        final_dir = output_dir if output_dir else f"~/.cache/huggingface/lerobot/{repo_id}"
        logging.info(f"Dataset saved locally at: {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ALOHA HDF5 dataset to LeRobot v3.0 format"
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing HDF5 files",
    )
    parser.add_argument(
        "--instruction-dir",
        type=Path,
        required=True,
        help="Directory containing instruction JSON files (episode{N}.json)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the converted dataset (default: ~/.cache/huggingface/lerobot/{repo-id})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="List of episode indices to convert (default: all)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload dataset to Hugging Face Hub",
    )

    args = parser.parse_args()

    port_aloha_hdf5(**vars(args))


if __name__ == "__main__":
    main()
