#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
Reflow (Rectified Flow) training for SmolVLA.

This module implements the Reflow training algorithm for SmolVLA, which straightens
the flow trajectories through iterative refinement with teacher-student distillation.

Paper: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
       https://arxiv.org/abs/2209.03003

Key components:
- SmolVLAReflowPolicy: Policy wrapper that handles teacher-student initialization
- VLAFlowMatchingReflow: Reflow-specific forward pass with ODE-based target generation

Usage:
    config = SmolVLAConfig(
        use_reflow=True,
        teacher_model_path="path/to/teacher",
        train_expert_only=True,  # Freeze VLM during reflow training
    )
    policy = make_policy(config)  # Automatically creates SmolVLAReflowPolicy
"""

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


# Import make_att_2d_masks from modeling_smolvla (it's a module-level function)
def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from 1D pad and attention masks.
    
    This is copied from modeling_smolvla.py to avoid circular import.
    """
    bsize, seqlen = pad_masks.shape
    pad_2d = pad_masks[:, None, :].expand(bsize, seqlen, seqlen)
    att_2d = att_masks[:, None, :].expand(bsize, seqlen, seqlen)
    att_2d_masks = pad_2d & att_2d
    return att_2d_masks


class SmolVLAReflowPolicy(SmolVLAPolicy):
    """SmolVLA Policy with Reflow (Rectified Flow) training.
    
    This class extends SmolVLAPolicy to support Reflow training, which uses a teacher
    model to generate straightened flow trajectories for more efficient training.
    
    The key difference from standard SmolVLAPolicy:
    1. Loads teacher model first (only loads VLM weights once)
    2. Creates student model without loading VLM weights
    3. Copies all weights from teacher to student
    4. Keeps teacher model cached for generating training targets
    
    Args:
        config: SmolVLAConfig with use_reflow=True and teacher_model_path set
    """
    
    def __init__(self, config: SmolVLAConfig):
        if not config.use_reflow:
            raise ValueError("SmolVLAReflowPolicy requires config.use_reflow=True")
        
        if not config.teacher_model_path:
            raise ValueError("SmolVLAReflowPolicy requires config.teacher_model_path to be set")
        
        # Ensure VLM is frozen during reflow training
        if not config.train_expert_only:
            print("[Reflow] Setting train_expert_only=True to freeze VLM")
            config.train_expert_only = True
        
        # IMPORTANT: Must call parent __init__ FIRST before assigning any modules
        # Initialize parent class (PreTrainedPolicy)
        super(SmolVLAPolicy, self).__init__(config)  # Call PreTrainedPolicy.__init__
        config.validate_features()
        self.config = config
        
        # Load teacher model (loads VLM weights once)
        print(f"[Reflow] Loading teacher model from {config.teacher_model_path}")
        self.teacher = SmolVLAPolicy.from_pretrained(config.teacher_model_path)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print("[Reflow] ✓ Teacher loaded and frozen")
        
        # Load student model from the same path (loads weights independently)
        print(f"[Reflow] Loading student model from {config.teacher_model_path}")
        student_policy = SmolVLAPolicy.from_pretrained(config.teacher_model_path)
        self.model = student_policy.model
        
        # Keep teacher on same device/dtype as student BEFORE binding to model.teacher
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.teacher = self.teacher.to(device=device, dtype=dtype)
        
        # Add teacher reference and bind reflow methods
        self.model.teacher = self.teacher
        self._bind_reflow_methods()
        print("[Reflow] ✓ Student loaded from pretrained path")
        
        self._log_trainable_params()
        self.reset()
    
    def _bind_reflow_methods(self):
        """Bind reflow methods to the student model."""
        import types
        
        # Bind generate_reflow_target method
        self.model.generate_reflow_target = types.MethodType(
            VLAFlowMatchingReflow.generate_reflow_target, self.model
        )
        
        # Bind forward_reflow method
        self.model.forward_reflow = types.MethodType(
            VLAFlowMatchingReflow.forward_reflow, self.model
        )
        
        # Replace forward method to use reflow
        original_forward = self.model.forward
        def forward_with_reflow(model_self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
            return model_self.forward_reflow(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        
        self.model.forward = types.MethodType(forward_with_reflow, self.model)
    
    def _log_trainable_params(self):
        """Log trainable parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Reflow] Total parameters: {total_params:,}")
        print(f"[Reflow] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model without reflow-specific config and teacher weights.
        
        This ensures saved checkpoints:
        1. Don't include use_reflow=True or teacher_model_path in config
        2. Can be loaded as standard SmolVLAPolicy for inference
        3. Have load_vlm_weights=True so VLM loads from checkpoint
        """
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

        # Save config without reflow artifacts
        config_dict = {
            k: v for k, v in vars(self.config).items()
            if k not in ('use_reflow', 'teacher_model_path')
        }
        
        # Ensure load_vlm_weights=True for inference
        config_dict['load_vlm_weights'] = True
        
        config_to_save = self.config.__class__(**config_dict)
        config_to_save._save_pretrained(save_directory)

        # Save model weights (only student, not teacher)
        state_dict = self.state_dict()
        self._save_to_safetensor(save_directory, state_dict, SAFETENSORS_SINGLE_FILE)


class VLAFlowMatchingReflow(VLAFlowMatching):
    """VLAFlowMatching with Reflow training support.
    
    This class extends VLAFlowMatching to implement the Reflow algorithm, which
    straightens flow trajectories through teacher-student distillation.
    
    Note: In practice, this class is not directly instantiated. Instead, we load
    a full VLAFlowMatching model and add a teacher attribute to it.
    """
    
    @torch.no_grad()
    def generate_reflow_target(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        """Generate X_1 from teacher model using ODE solver (Euler method).

        This implements the data pair generation step in Reflow (Algorithm 1):
        X_0 ~ N(0, 1) (noise)
        X_1 = ODE[v_k](X_0 | T) = X_0 + ∫_0^1 v_k(X_t, t | T) dt

        Args:
            images, img_masks, lang_tokens, lang_masks, state: Input observations
            noise: X_0, sampled from N(0, 1)

        Returns:
            X_1: Generated actions from teacher model via ODE integration
        """
        bsize = noise.shape[0]
        device = noise.device

        # Get prefix embeddings and cache (shared across denoising steps)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.teacher.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Create position_ids and attention_mask from prefix masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.teacher.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
            fill_kv_cache=True,
        )

        # ODE integration using Euler method
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps
        x_t = noise  # Start from X_0 = noise

        for step in range(num_steps):
            t = step * dt
            time = torch.full((bsize,), t, device=device, dtype=torch.float32)

            # Compute v_k(X_t, t | T) using teacher model
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.teacher.model.embed_suffix(x_t, time)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (_, suffix_out), _ = self.teacher.model.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )

            v_t = self.teacher.model.action_out_proj(suffix_out)

            # Euler step: X_{t+dt} = X_t + dt * v_t
            x_t = x_t + dt * v_t

        return x_t  # X_1

    def forward_reflow(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Reflow training forward pass (Rectified Flow).

        Implements Equation 5 from Reflow paper:
        L_reflow = E[ ∫ ||(X_1 - X_0) - v_{k+1}(X_t, t)||^2 dt ]

        Where:
        - X_0 ~ N(0, 1): noise
        - X_1 = ODE[v_k](X_0|T): generated by teacher model v_k
        - X_t = (1-t)*X_0 + t*X_1: linear interpolation
        - (X_1 - X_0): straight-line velocity (target)
        - v_{k+1}: student model (being trained)

        The key difference from standard Flow Matching:
        - Standard FM: u_t = noise - actions (curved trajectory)
        - Reflow: u_t = X_1 - X_0 (straightened trajectory)
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)  # X_0

        # Generate X_1 using teacher model via ODE
        X_1 = self.generate_reflow_target(images, img_masks, lang_tokens, lang_masks, state, noise)
        X_0 = noise

        # Sample timestep
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Linear interpolation: X_t = (1-t)*X_0 + t*X_1
        time_expanded = time[:, None, None]
        X_t = (1 - time_expanded) * X_0 + time_expanded * X_1

        # Target: straight-line velocity from X_0 to X_1
        u_t = X_1 - X_0

        # Compute student model prediction v_{k+1}(X_t, t)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(X_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        v_t_pred = self.action_out_proj(suffix_out)

        # Reflow loss: ||u_t - v_t_pred||^2
        losses = (u_t - v_t_pred) ** 2

        return losses

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Forward pass with Reflow training."""
        return self.forward_reflow(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
