#!/usr/bin/env python3
"""
Script to reconvert corrupted video episodes in the LeRobot dataset.
This script fixes corrupted MP4 files by re-encoding them from the original HDF5 data.
"""

import os
import cv2
import h5py
import numpy as np
from pathlib import Path
import argparse
import tqdm
import av
from PIL import Image
import tempfile
import shutil

def get_cameras(hdf5_file: Path) -> list[str]:
    with h5py.File(hdf5_file, "r") as ep:
        return [key for key in ep["/observation"].keys() if "depth" not in key]

def load_raw_images_for_camera(ep: h5py.File, camera: str) -> np.ndarray:
    """Load images for a specific camera from HDF5 file."""
    uncompressed = ep[f"/observation/{camera}/rgb"].ndim == 4

    if uncompressed:
        imgs_array = ep[f"/observation/{camera}/rgb"][:]
    else:
        imgs_array = []
        for data in ep[f"/observation/{camera}/rgb"]:
            data = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs_array.append(img)

    imgs_array = [cv2.resize(img, (640, 480)) for img in imgs_array]
    imgs_array = np.stack(imgs_array, axis=0)
    return imgs_array

def reencode_video(hdf5_path: Path, output_mp4_path: Path, camera: str = "right_camera"):
    """Re-encode a single video from HDF5 to MP4 using the same method as LeRobot."""
    with h5py.File(hdf5_path, "r") as ep:
        imgs_array = load_raw_images_for_camera(ep, camera)

    # Use LeRobot's encoding approach directly from numpy arrays
    encode_video_frames_direct(imgs_array, output_mp4_path, fps=30)

def encode_video_frames_direct(
    imgs_array: np.ndarray,
    video_path: Path,
    fps: int = 30,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fast_decode: int = 0,
):
    """Encode video frames directly from numpy array using the same parameters as LeRobot."""
    video_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames, height, width, channels = imgs_array.shape

    # Define video codec options
    video_options = {"g": str(g), "crf": str(crf)}
    
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    av.logging.set_level(av.logging.ERROR)

    # Create and open output file
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for frame in imgs_array:
            # Convert numpy array to PIL Image then to AV frame
            pil_image = Image.fromarray(frame)
            input_frame = av.VideoFrame.from_image(pil_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")
    
    print(f"Re-encoded video: {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Reconvert corrupted video episodes")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobot dataset directory")
    parser.add_argument("--corrupted-episodes", type=int, nargs='+', required=True,
                       help="List of corrupted episode indices (e.g., 104 339 428)")
    parser.add_argument("--camera", type=str, default="right_camera",
                       help="Camera name that has corrupted videos")

    args = parser.parse_args()

    # Find all HDF5 files and sort by episode number
    import fnmatch
    import re

    hdf5_files = []
    for root, _, files in os.walk(args.raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)
    hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"episode(\d+)\.hdf5", os.path.basename(x)).group(1)))

    print(f"Found {len(hdf5_files)} HDF5 files")

    # Process each corrupted episode
    for ep_idx in tqdm.tqdm(args.corrupted_episodes, desc="Re-encoding corrupted episodes"):
        if ep_idx >= len(hdf5_files):
            print(f"Warning: Episode index {ep_idx} is out of range (max: {len(hdf5_files)-1})")
            continue

        hdf5_path = Path(hdf5_files[ep_idx])
        episode_num = f"{ep_idx:06d}"
        output_mp4_path = args.dataset_dir / "videos" / "chunk-000" / f"observation.images.{args.camera}" / f"episode_{episode_num}.mp4"

        print(f"Processing episode {ep_idx} -> {output_mp4_path}")

        # Backup the corrupted file (optional)
        if output_mp4_path.exists():
            backup_path = output_mp4_path.with_suffix('.mp4.backup')
            if not backup_path.exists():
                output_mp4_path.rename(backup_path)
                print(f"Backed up corrupted file to {backup_path}")

        # Re-encode the video
        reencode_video(hdf5_path, output_mp4_path, args.camera)

if __name__ == "__main__":
    main()