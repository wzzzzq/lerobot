#!/usr/bin/env python

"""
LeRobot Dataset Integrity Checker

This script checks the integrity of a LeRobot v3.0 dataset by:
1. Verifying all video files can be decoded completely
2. Checking all parquet data files are readable
3. Validating metadata consistency
4. Checking episode completeness

Usage:
    python examples/check_dataset_integrity.py \
        --dataset-dir /path/to/dataset \
        --check-videos \
        --check-data \
        --check-metadata
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Try to import av for video checking (optional)
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    print("⚠️  Warning: PyAV not installed. Video checking will be skipped.")
    print("   Install with: pip install av")


class DatasetIntegrityChecker:
    """Check integrity of a LeRobot dataset."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_videos": 0,
            "corrupted_videos": 0,
            "total_data_files": 0,
            "corrupted_data_files": 0,
            "total_episodes": 0,
        }

    def add_error(self, message: str, file_path: Path | None = None):
        """Record an error."""
        error_msg = f"❌ ERROR: {message}"
        if file_path:
            error_msg += f"\n   File: {file_path}"
        self.errors.append(error_msg)
        print(error_msg)

    def add_warning(self, message: str, file_path: Path | None = None):
        """Record a warning."""
        warning_msg = f"⚠️  WARNING: {message}"
        if file_path:
            warning_msg += f"\n   File: {file_path}"
        self.warnings.append(warning_msg)
        print(warning_msg)

    def check_directory_structure(self) -> bool:
        """Check if the dataset has the expected directory structure."""
        print("\n🔍 Checking directory structure...")

        required_files = [
            "meta/info.json",
        ]

        required_dirs = [
            "data",
            "meta",
            "meta/episodes",
            "meta/tasks",
        ]

        all_exist = True

        # Check required files
        for file_path in required_files:
            full_path = self.dataset_dir / file_path
            if not full_path.exists():
                self.add_error(f"Required file missing: {file_path}")
                all_exist = False

        # Check required directories
        for dir_path in required_dirs:
            full_path = self.dataset_dir / dir_path
            if not full_path.exists():
                self.add_error(f"Required directory missing: {dir_path}")
                all_exist = False

        if all_exist:
            print("✅ Directory structure is valid")
        return all_exist

    def load_info(self) -> dict | None:
        """Load and validate info.json."""
        print("\n🔍 Checking meta/info.json...")

        info_path = self.dataset_dir / "meta" / "info.json"
        try:
            with open(info_path) as f:
                info = json.load(f)

            # Check required fields
            required_fields = [
                "codebase_version",
                "robot_type",
                "total_episodes",
                "total_frames",
                "fps",
                "features",
            ]

            for field in required_fields:
                if field not in info:
                    self.add_error(f"Required field missing in info.json: {field}")
                    return None

            print(f"✅ Dataset info:")
            print(f"   - Robot type: {info['robot_type']}")
            print(f"   - Total episodes: {info['total_episodes']}")
            print(f"   - Total frames: {info['total_frames']}")
            print(f"   - FPS: {info['fps']}")
            print(f"   - Codebase version: {info['codebase_version']}")

            self.stats["total_episodes"] = info["total_episodes"]
            return info

        except json.JSONDecodeError as e:
            self.add_error(f"Failed to parse info.json: {e}", info_path)
            return None
        except Exception as e:
            self.add_error(f"Failed to read info.json: {e}", info_path)
            return None

    def check_videos(self) -> bool:
        """Check integrity of all video files."""
        if not AV_AVAILABLE:
            self.add_warning("PyAV not available, skipping video checks")
            return True

        print("\n🔍 Checking video files...")

        videos_dir = self.dataset_dir / "videos"
        if not videos_dir.exists():
            print("ℹ️  No videos directory found (dataset may not have videos)")
            return True

        # Find all video files
        video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".webm")
        video_files = []
        for ext in video_extensions:
            video_files.extend(videos_dir.rglob(f"*{ext}"))

        if not video_files:
            print("ℹ️  No video files found")
            return True

        self.stats["total_videos"] = len(video_files)
        print(f"📂 Found {len(video_files)} video files")

        corrupted_videos = []

        # Check each video file
        for video_path in tqdm(video_files, desc="Checking videos"):
            try:
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    frame_count = 0
                    for frame in container.decode(stream):
                        frame_count += 1
                    # Verify we got at least one frame
                    if frame_count == 0:
                        corrupted_videos.append((video_path, "No frames found"))
            except Exception as e:
                error_info = f"{type(e).__name__}: {str(e)}"
                corrupted_videos.append((video_path, error_info))
                tqdm.write(f"❌ Corrupted video: {video_path.relative_to(self.dataset_dir)}")

        self.stats["corrupted_videos"] = len(corrupted_videos)

        if corrupted_videos:
            print(f"\n❌ Found {len(corrupted_videos)} corrupted video files:")
            for path, error in corrupted_videos:
                print(f"   - {path.relative_to(self.dataset_dir)}")
                print(f"     Error: {error}")
            return False
        else:
            print("✅ All video files are valid")
            return True

    def check_data_files(self) -> bool:
        """Check integrity of all parquet data files."""
        print("\n🔍 Checking data files...")

        data_dir = self.dataset_dir / "data"
        if not data_dir.exists():
            self.add_error("Data directory not found")
            return False

        # Find all parquet files
        parquet_files = list(data_dir.rglob("*.parquet"))

        if not parquet_files:
            self.add_error("No parquet data files found")
            return False

        self.stats["total_data_files"] = len(parquet_files)
        print(f"📂 Found {len(parquet_files)} data files")

        corrupted_files = []

        # Check each parquet file
        for parquet_path in tqdm(parquet_files, desc="Checking data files"):
            try:
                df = pd.read_parquet(parquet_path)
                # Verify we have data
                if len(df) == 0:
                    corrupted_files.append((parquet_path, "Empty dataframe"))
            except Exception as e:
                error_info = f"{type(e).__name__}: {str(e)}"
                corrupted_files.append((parquet_path, error_info))
                tqdm.write(f"❌ Corrupted data file: {parquet_path.relative_to(self.dataset_dir)}")

        self.stats["corrupted_data_files"] = len(corrupted_files)

        if corrupted_files:
            print(f"\n❌ Found {len(corrupted_files)} corrupted data files:")
            for path, error in corrupted_files:
                print(f"   - {path.relative_to(self.dataset_dir)}")
                print(f"     Error: {error}")
            return False
        else:
            print("✅ All data files are valid")
            return True

    def check_episodes_metadata(self) -> bool:
        """Check episode metadata consistency."""
        print("\n🔍 Checking episode metadata...")

        episodes_dir = self.dataset_dir / "meta" / "episodes"
        if not episodes_dir.exists():
            self.add_error("Episodes metadata directory not found")
            return False

        # Find all episode parquet files
        episode_files = list(episodes_dir.rglob("*.parquet"))

        if not episode_files:
            self.add_error("No episode metadata files found")
            return False

        print(f"📂 Found {len(episode_files)} episode metadata files")

        all_episodes = []

        # Read all episode metadata
        for ep_file in episode_files:
            try:
                df = pd.read_parquet(ep_file)
                all_episodes.append(df)
            except Exception as e:
                self.add_error(f"Failed to read episode metadata: {e}", ep_file)
                return False

        # Concatenate all episodes
        episodes_df = pd.concat(all_episodes, ignore_index=True)

        # Check episode indices are continuous
        episode_indices = sorted(episodes_df["episode_index"].tolist())
        expected_indices = list(range(len(episode_indices)))

        if episode_indices != expected_indices:
            self.add_error(
                f"Episode indices are not continuous. "
                f"Expected {len(expected_indices)} episodes (0-{len(expected_indices)-1}), "
                f"but found: {episode_indices}"
            )
            return False

        # Check against info.json
        if self.stats["total_episodes"] != len(episode_indices):
            self.add_error(
                f"Episode count mismatch. "
                f"info.json says {self.stats['total_episodes']}, "
                f"but found {len(episode_indices)} episodes in metadata"
            )
            return False

        print(f"✅ Found {len(episode_indices)} valid episodes")
        return True

    def check_tasks_metadata(self) -> bool:
        """Check task metadata."""
        print("\n🔍 Checking task metadata...")

        tasks_dir = self.dataset_dir / "meta" / "tasks"
        if not tasks_dir.exists():
            self.add_warning("Tasks metadata directory not found")
            return True

        # Find all task parquet files
        task_files = list(tasks_dir.rglob("*.parquet"))

        if not task_files:
            self.add_warning("No task metadata files found")
            return True

        # Read all task metadata
        all_tasks = []
        for task_file in task_files:
            try:
                df = pd.read_parquet(task_file)
                all_tasks.append(df)
            except Exception as e:
                self.add_error(f"Failed to read task metadata: {e}", task_file)
                return False

        tasks_df = pd.concat(all_tasks, ignore_index=True)
        print(f"✅ Found {len(tasks_df)} tasks")
        return True

    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("📊 INTEGRITY CHECK SUMMARY")
        print("=" * 60)

        print("\n📈 Statistics:")
        print(f"   Total episodes: {self.stats['total_episodes']}")
        print(f"   Total videos: {self.stats['total_videos']}")
        print(f"   Corrupted videos: {self.stats['corrupted_videos']}")
        print(f"   Total data files: {self.stats['total_data_files']}")
        print(f"   Corrupted data files: {self.stats['corrupted_data_files']}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   {warning}")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   {error}")
            print("\n🚨 Dataset has integrity issues!")
            return False
        else:
            print("\n✅ Dataset integrity check PASSED!")
            print("🎉 All checks completed successfully!")
            return True

    def run_checks(
        self,
        check_videos: bool = True,
        check_data: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """Run all integrity checks."""
        print("=" * 60)
        print(f"🔍 Checking dataset: {self.dataset_dir}")
        print("=" * 60)

        # Check directory structure
        if not self.check_directory_structure():
            return False

        # Load info
        info = self.load_info()
        if info is None:
            return False

        all_passed = True

        # Check videos
        if check_videos:
            if not self.check_videos():
                all_passed = False

        # Check data files
        if check_data:
            if not self.check_data_files():
                all_passed = False

        # Check metadata
        if check_metadata:
            if not self.check_episodes_metadata():
                all_passed = False
            if not self.check_tasks_metadata():
                all_passed = False

        # Print summary
        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Check integrity of a LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset directory",
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        default=True,
        help="Check video file integrity (requires PyAV)",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        default=True,
        help="Check parquet data file integrity",
    )
    parser.add_argument(
        "--check-metadata",
        action="store_true",
        default=True,
        help="Check metadata consistency",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video checks",
    )

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"❌ Error: Dataset directory does not exist: {args.dataset_dir}")
        sys.exit(1)

    # Create checker
    checker = DatasetIntegrityChecker(args.dataset_dir)

    # Run checks
    check_videos = args.check_videos and not args.skip_videos
    success = checker.run_checks(
        check_videos=check_videos,
        check_data=args.check_data,
        check_metadata=args.check_metadata,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
