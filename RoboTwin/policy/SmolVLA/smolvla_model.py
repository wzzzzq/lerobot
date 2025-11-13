from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
import torch
from torch import Tensor
import numpy as np

class SmolVLA(SmolVLAPolicy):
    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.observation_window = None  # 新增属性
        self.instruction = None
        self.tokenized_instruction = None

        # Access tokenizer from the VLM model
        self.tokenizer = self.model.vlm_with_expert.processor.tokenizer
        self.max_length = self.config.tokenizer_max_length

    def set_language(self, instruction):
        self.instruction = instruction

        # Get device from model parameters
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Tokenize the instruction
        tokenized = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device and store
        self.tokenized_instruction = {
            "input_ids": tokenized["input_ids"].to(device),
            "attention_mask": tokenized["attention_mask"].to(device),
        }
    def update_observation_window(self, img_arr, state):
        """
        Dynamically handle different number of cameras.
        img_arr should be a list/array of images, order matches camera names in camera_names.
        """
        if self.tokenized_instruction is None:
            raise ValueError("Must call set_language() before update_observation_window()")

        def prepare_img(img):
            img = np.transpose(img, (2, 0, 1))
            img = img[np.newaxis, ...]
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(device)

        # Build observation window with tokenized language
        self.observation_window = {
            "observation.state": state_tensor,
            OBS_LANGUAGE_TOKENS: self.tokenized_instruction["input_ids"],
            OBS_LANGUAGE_ATTENTION_MASK: self.tokenized_instruction["attention_mask"],
        }
        
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
        
        # Add each camera to observation window
        for i, camera_name in enumerate(camera_names):
            if i < len(img_arr):
                key = f"observation.images.{camera_name}"
                self.observation_window[key] = prepare_img(img_arr[i]).to(device)
    def get_action(self):
        assert self.observation_window is not None, "Update observation_window first!"
        
        action_tensor = self.select_action(self.observation_window)
        
        action_numpy = action_tensor.cpu().numpy()
        action_numpy = action_numpy.squeeze(0)
        
        return action_numpy
    