from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction
import torch
from torch import Tensor
import numpy as np
from pathlib import Path

class SmolVLA(SmolVLAPolicy):
    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.observation_window = None  # 新增属性
        self.instruction = None
        self.preprocessor = None
        self.postprocessor = None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Override to load preprocessor and postprocessor."""
        # Load policy using parent method
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)

        # Load preprocessor and postprocessor
        policy.preprocessor, policy.postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=pretrained_name_or_path,
        )

        return policy

    def set_language(self, instruction):
        """Set language instruction - will be processed by preprocessor."""
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        """
        Build observation window using preprocessor for proper normalization.

        Args:
            img_arr: List of images in HWC format, range [0, 255], order matches camera_names
            state: Robot joint state vector (raw values, will be normalized by preprocessor)
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not loaded. Make sure to load the model using "
                "SmolVLA.from_pretrained() to include preprocessor and postprocessor."
            )

        if self.instruction is None:
            raise ValueError("Must call set_language() before update_observation_window()")

        # Prepare raw observation dict (before preprocessing)
        def prepare_img(img):
            """Convert HWC [0,255] to CHW [0,1] format."""
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = img.astype(np.float32) / 255.0  # [0,255] -> [0,1]
            return torch.from_numpy(img)

        # Get camera names from model config or use default order
        if hasattr(self.config, 'camera_names'):
            camera_names = self.config.camera_names
        else:
            # Default order: try to match common patterns
            if len(img_arr) == 4:
                camera_names = ["head_camera", "right_camera", "left_camera", "front_camera"]
            elif len(img_arr) == 3:
                camera_names = ["head_camera", "left_camera", "right_camera"]
            else:
                # Fallback: create generic names
                camera_names = [f"camera_{i}" for i in range(len(img_arr))]

        # Build raw observation (unnormalized state, [0,1] images)
        raw_observation = {
            "observation": {
                "state": torch.from_numpy(np.array(state, dtype=np.float32)),
            }
        }

        # Add camera images
        for i, camera_name in enumerate(camera_names):
            if i < len(img_arr):
                key = f"observation.images.{camera_name}"
                raw_observation[key] = prepare_img(img_arr[i])

        # Add language instruction
        raw_observation["task"] = self.instruction

        # Apply preprocessor (normalizes state, tokenizes language, moves to device)
        self.observation_window = self.preprocessor(raw_observation)

    def get_action(self):
        """Get action and apply postprocessor for denormalization."""
        if self.observation_window is None:
            raise ValueError("Must call update_observation_window() first!")

        if self.postprocessor is None:
            raise ValueError(
                "Postprocessor not loaded. Make sure to load the model using "
                "SmolVLA.from_pretrained() to include preprocessor and postprocessor."
            )

        # Get normalized action from policy
        action_tensor = self.select_action(self.observation_window)

        # Apply postprocessor to denormalize action (CRITICAL!)
        policy_action = PolicyAction(action=action_tensor)
        denormalized_action = self.postprocessor(policy_action)

        # Extract action numpy array
        action_numpy = denormalized_action.action.cpu().numpy().squeeze(0)

        return action_numpy

    def reset(self):
        """Reset policy state including observation window."""
        super().reset()  # Reset parent's action queue
        self.observation_window = None
        self.instruction = None
