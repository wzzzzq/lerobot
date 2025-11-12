#!/usr/bin/env python3
"""
Script to detect and fix corrupted video files in LeRobot datasets.

This script:
1. Scans dataset for corrupted MP4 videos
2. Identifies which episodes need fixing
3. Re-encodes them from original HDF5 files

Usage:
    python examples/reconvert_corrupted_videos.py \\
        --dataset-dir /path/to/dataset \\
        --hdf5-dir /path/to/hdf5_files
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def detect_corrupted_videos(dataset_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """Scan dataset and detect corrupted videos.

    Returns:
        Dict mapping camera_key -> [(episode_idx, video_path), ...]
    """
    print("🔍 Scanning dataset for corrupted videos...")

    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        print("❌ No videos directory found")
        return {}

    corrupted_by_camera = {}

    # Get all camera directories
    for camera_dir in sorted(videos_dir.iterdir()):
        if not camera_dir.is_dir():
            continue

        camera_key = camera_dir.name
        print(f"\n📹 Checking {camera_key}...")

        # Find all video files for this camera
        video_files = list(camera_dir.rglob("*.mp4"))

        corrupted = []
        for video_path in tqdm(video_files, desc=f"  Checking videos", leave=False):
            try:
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    frame_count = 0
                    for frame in container.decode(stream):
                        frame_count += 1

                    if frame_count == 0:
                        corrupted.append(video_path)
            except Exception:
                corrupted.append(video_path)

        if corrupted:
            print(f"  ❌ Found {len(corrupted)} corrupted videos")
            # Try to extract episode indices from video paths
            for video_path in corrupted:
                # Try to find episode index from filename or parent directory
                # Pattern: file-XXX.mp4 or episode-XXXXXX.mp4
                match = re.search(r'file-(\d+)\.mp4', video_path.name)
                if match:
                    ep_idx = int(match.group(1))
                else:
                    match = re.search(r'episode-(\d+)\.mp4', video_path.name)
                    if match:
                        ep_idx = int(match.group(1))
                    else:
                        print(f"  ⚠️  Cannot extract episode index from: {video_path.name}")
                        continue

                if camera_key not in corrupted_by_camera:
                    corrupted_by_camera[camera_key] = []
                corrupted_by_camera[camera_key].append((ep_idx, video_path))
        else:
            print(f"  ✅ All videos OK")

    return corrupted_by_camera


def find_hdf5_files(hdf5_dir: Path) -> List[Path]:
    """Find and sort all HDF5 files."""
    hdf5_files = []

    for root, dirs, files in os.walk(hdf5_dir):
        for filename in files:
            if filename.endswith('.hdf5'):
                hdf5_files.append(Path(root) / filename)

    # Sort by episode number
    def get_episode_num(path):
        match = re.search(r'episode[_-]?(\d+)', path.stem)
        if match:
            return int(match.group(1))
        # Try to extract from directory structure
        match = re.search(r'episode[_-]?(\d+)', str(path))
        if match:
            return int(match.group(1))
        return 0

    hdf5_files = sorted(hdf5_files, key=get_episode_num)
    return hdf5_files


def get_camera_name_from_key(camera_key: str) -> str:
    """Extract camera name from full key.

    e.g., 'observation.images.cam_left_wrist' -> 'cam_left_wrist'
    """
    parts = camera_key.split('.')
    return parts[-1] if parts else camera_key


def load_images_from_hdf5(hdf5_file: Path, camera: str) -> np.ndarray:
    """Load images from HDF5 file for a specific camera."""
    with h5py.File(hdf5_file, "r") as ep:
        # Try different possible paths
        possible_paths = [
            f"/observation/{camera}/rgb",
            f"/observation/images/{camera}",
            f"/observations/images/{camera}",
        ]

        rgb_data = None
        for path in possible_paths:
            try:
                if path in ep:
                    rgb_data = ep[path]
                    break
            except:
                continue

        if rgb_data is None:
            # Try to find any matching camera
            obs = ep["/observation"] if "/observation" in ep else ep["/observations"]
            for key in obs.keys():
                if camera in key:
                    rgb_data = obs[key]["rgb"] if "rgb" in obs[key] else obs[key]
                    break

        if rgb_data is None:
            raise KeyError(f"Cannot find camera '{camera}' in HDF5. Available: {list(ep['/observation'].keys())}")

        # Check if compressed
        uncompressed = rgb_data.ndim == 4

        if uncompressed:
            imgs_array = rgb_data[:]
        else:
            # Decode compressed images
            imgs_array = []
            for data in rgb_data:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_array.append(img)
            imgs_array = np.stack(imgs_array, axis=0)

        # Resize to standard size
        imgs_array = np.stack([
            cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            for img in imgs_array
        ], axis=0)

        return imgs_array


def encode_video(imgs_array: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Encode images to video file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames, height, width, channels = imgs_array.shape

    # Video encoding options (same as LeRobot)
    video_options = {
        "g": "2",      # GOP size
        "crf": "23",   # Quality
    }

    av.logging.set_level(av.logging.ERROR)

    with av.open(str(output_path), "w") as output:
        stream = output.add_stream("libx264", fps, options=video_options)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height

        for frame in imgs_array:
            pil_image = Image.fromarray(frame.astype(np.uint8))
            av_frame = av.VideoFrame.from_image(pil_image)

            for packet in stream.encode(av_frame):
                output.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            output.mux(packet)

    if not output_path.exists():
        raise OSError(f"Video encoding failed: {output_path}")


def fix_corrupted_video(
    hdf5_path: Path,
    video_path: Path,
    camera_name: str,
    fps: int,
) -> None:
    """Fix a single corrupted video by re-encoding from HDF5."""
    # Backup corrupted file
    backup_path = video_path.with_suffix('.mp4.corrupted')
    if not backup_path.exists():
        shutil.copy2(video_path, backup_path)

    # Load images from HDF5
    imgs_array = load_images_from_hdf5(hdf5_path, camera_name)

    # Re-encode video
    encode_video(imgs_array, video_path, fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and fix corrupted videos in LeRobot datasets",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        required=True,
        help="Path to directory containing HDF5 source files",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    if not args.hdf5_dir.exists():
        print(f"❌ HDF5 directory not found: {args.hdf5_dir}")
        sys.exit(1)

    # Load dataset info
    info_path = args.dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"❌ Dataset info not found: {info_path}")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    fps = info.get("fps", 30)

    # Step 1: Detect corrupted videos
    corrupted_by_camera = detect_corrupted_videos(args.dataset_dir)

    if not corrupted_by_camera:
        print("\n✅ No corrupted videos found! Dataset is healthy.")
        return

    # Summary
    total_corrupted = sum(len(videos) for videos in corrupted_by_camera.values())
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total corrupted videos: {total_corrupted}")
    for camera_key, videos in corrupted_by_camera.items():
        episodes = sorted(set(ep_idx for ep_idx, _ in videos))
        print(f"  {camera_key}: {len(videos)} videos, episodes {episodes}")

    # Step 2: Find HDF5 files
    print(f"\n{'='*60}")
    print("📂 Loading HDF5 files...")
    hdf5_files = find_hdf5_files(args.hdf5_dir)
    print(f"Found {len(hdf5_files)} HDF5 files")

    if not hdf5_files:
        print("❌ No HDF5 files found!")
        sys.exit(1)

    # Step 3: Fix corrupted videos
    print(f"\n{'='*60}")
    print("🔧 Fixing corrupted videos...")
    print(f"{'='*60}\n")

    fixed_count = 0
    failed_count = 0

    for camera_key, corrupted_videos in corrupted_by_camera.items():
        camera_name = get_camera_name_from_key(camera_key)
        print(f"\n📹 Processing {camera_key} ({camera_name})")

        for ep_idx, video_path in tqdm(corrupted_videos, desc=f"  Fixing videos"):
            if ep_idx >= len(hdf5_files):
                print(f"  ⚠️  Episode {ep_idx} out of range (max: {len(hdf5_files)-1})")
                failed_count += 1
                continue

            hdf5_path = hdf5_files[ep_idx]

            try:
                fix_corrupted_video(hdf5_path, video_path, camera_name, fps)
                fixed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to fix episode {ep_idx}: {e}")
                failed_count += 1

    # Final summary
    print(f"\n{'='*60}")
    print("✅ COMPLETE")
    print(f"{'='*60}")
    print(f"Fixed: {fixed_count} videos")
    if failed_count > 0:
        print(f"Failed: {failed_count} videos")
    print("\nCorrupted videos backed up with .corrupted extension")


if __name__ == "__main__":
    main()
