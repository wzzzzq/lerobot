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
Reflow training script for SmolVLA.

This script extends the standard lerobot_train.py to support Reflow (Rectified Flow) training.
It loads a teacher model AND initializes the student from the same checkpoint.

Key differences from standard training:
1. Student model is initialized from teacher_model_path (NOT created fresh)
2. Teacher model is loaded from teacher_model_path and attached to student
3. Sets policy.model.training_mode = "reflow" (runtime state)
4. Uses smaller learning rate (typically 2e-5 vs 1e-4)

Important: Both teacher and student load from the SAME checkpoint path.
This is correct for reflow - the student starts with teacher's weights,
then is fine-tuned to straighten the trajectories.

Usage:
    python lerobot_train_reflow.py \\
        --policy.type=smolvla \\
        --policy.teacher_model_path=/path/to/teacher \\
        --policy.optimizer_lr=2e-5 \\
        --dataset.repo_id=your_dataset \\
        --steps=20000
"""

import logging
import sys
import os

# Import the main training function from lerobot_train
sys.path.insert(0, os.path.dirname(__file__))
from lerobot_train import *  # noqa: F403, E402


def setup_reflow_training(cfg, ds_meta):
    """Setup reflow training by loading both teacher and student from teacher checkpoint.

    In reflow training, the student should be initialized from the teacher's weights,
    then fine-tuned to straighten the trajectories.

    Note: training_mode is NOT in config - it's a runtime state set by this function.

    Args:
        cfg: Training configuration
        ds_meta: Dataset metadata

    Returns:
        Student policy initialized from teacher checkpoint with teacher attached
    """
    if cfg.policy.type != "smolvla":
        raise ValueError("Reflow training is only supported for smolvla policy")

    teacher_path = getattr(cfg.policy, "teacher_model_path", None)
    if not teacher_path:
        raise ValueError(
            "teacher_model_path must be specified for reflow training. "
            "Use --policy.teacher_model_path=/path/to/teacher"
        )

    logging.info(f"[Reflow] Loading teacher model from {teacher_path}")

    # Import SmolVLAPolicy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # Load teacher model
    teacher = SmolVLAPolicy.from_pretrained(teacher_path)
    teacher.eval()

    # Freeze all teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    logging.info("[Reflow] ✓ Teacher model loaded")

    # Initialize student from the same checkpoint (this is key for reflow!)
    # The student starts with the same weights as the teacher, then is fine-tuned
    logging.info(f"[Reflow] Initializing student model from {teacher_path}")
    student = SmolVLAPolicy.from_pretrained(teacher_path)

    # Apply training configuration to student
    # These settings may differ from the original training
    if hasattr(cfg.policy, "freeze_vision_encoder"):
        # Re-apply freezing settings based on current config
        if cfg.policy.freeze_vision_encoder:
            logging.info("[Reflow] Freezing vision encoder")
            for param in student.model.vlm_with_expert.vlm.vision_tower.parameters():
                param.requires_grad = False
            for param in student.model.vlm_with_expert.vlm.vision_encoder.parameters():
                param.requires_grad = False

    if hasattr(cfg.policy, "train_expert_only") and cfg.policy.train_expert_only:
        logging.info("[Reflow] Training expert only (freezing VLM language model)")
        for param in student.model.vlm_with_expert.vlm.language_model.parameters():
            param.requires_grad = False

    # Attach teacher to student model
    student.model.teacher = teacher

    # Set training mode to reflow (runtime state, not in config)
    student.model.training_mode = "reflow"

    logging.info("[Reflow] ✓ Student initialized from teacher checkpoint")
    logging.info(f"[Reflow] ✓ Training mode: {student.model.training_mode}")

    # Log parameter counts
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logging.info(f"[Reflow] Total parameters: {total_params:,}")
    logging.info(f"[Reflow] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return student


def main():
    """Main training function with reflow support."""
    # Parse config
    cfg = parser.parse_args_to_cfg()  # noqa: F405
    init_logging()  # noqa: F405

    # Set seed
    set_seed(cfg.seed)  # noqa: F405

    # Log config
    logging.info(f"Training with config:\n{pformat(cfg)}")  # noqa: F405

    # Make dataset
    logging.info("Making dataset...")
    ds_meta, train_dataloader = make_dataset(cfg)  # noqa: F405

    # Setup reflow training (loads both teacher and student from teacher checkpoint)
    # NOTE: Student is initialized from teacher's weights, NOT created fresh
    policy = setup_reflow_training(cfg, ds_meta)
    pre_processor, post_processor = make_pre_post_processors(cfg.policy)  # noqa: F405

    # Make optimizer and scheduler
    logging.info("Making optimizer and scheduler...")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg.policy, policy)  # noqa: F405

    # Setup accelerator
    accelerator = Accelerator(  # noqa: F405
        project_dir=cfg.output_dir,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.wandb.enable else None,
    )

    # Prepare for distributed training
    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize WandB if enabled
    if cfg.wandb.enable:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=dict(cfg),
            init_kwargs={
                "wandb": {
                    "entity": cfg.wandb.entity,
                    "mode": cfg.wandb.mode,
                    "notes": cfg.wandb.notes,
                }
            },
        )

    # Training loop
    logging.info("Starting training...")

    # Get data iterator
    data_iter = cycle(train_dataloader)  # noqa: F405

    # Metrics tracker
    train_metrics = MetricsTracker()  # noqa: F405

    # Training step
    step = 0
    while step < cfg.steps:
        # Get batch
        batch = next(data_iter)

        # Update policy
        train_metrics, output_dict = update_policy(  # noqa: F405
            train_metrics=train_metrics,
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=cfg.policy.optimizer_grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1

        # Log metrics
        if step % cfg.log_freq == 0:
            metrics = train_metrics.compute_metrics()
            logging.info(
                f"Step {step}/{cfg.steps} | "
                f"Loss: {metrics.get('loss', 0):.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if cfg.wandb.enable:
                accelerator.log(metrics, step=step)

        # Save checkpoint
        if step % cfg.save_freq == 0 or step == cfg.steps:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, step)  # noqa: F405
            logging.info(f"Saving checkpoint to {checkpoint_dir}")

            # Unwrap policy from accelerator
            unwrapped_policy = accelerator.unwrap_model(policy)

            # Save checkpoint
            save_checkpoint(  # noqa: F405
                policy=unwrapped_policy,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step=step,
                output_dir=checkpoint_dir,
            )

            update_last_checkpoint(cfg.output_dir, step)  # noqa: F405

    logging.info("Training completed!")

    # Finish WandB
    if cfg.wandb.enable:
        accelerator.end_training()


if __name__ == "__main__":
    main()
