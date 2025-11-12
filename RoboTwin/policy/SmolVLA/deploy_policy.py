# deploy_policy.py for SmolVLA
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from typing import Any, Dict
from smolvla_model import SmolVLA

model: SmolVLA = None
device: str = "cpu"

def get_model(usr_args: Dict) -> SmolVLA:
    global model, device
    
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SmolVLA model on device: {device}")
    
    # Build full policy path: policy_path/ckpt_setting/pretrained_model
    policy_base_path = usr_args["policy_path"]
    ckpt_setting = usr_args.get("ckpt_setting", "last")
    policy_path = os.path.join(policy_base_path, ckpt_setting, "pretrained_model")
    
    print(f"Loading from checkpoint: {policy_path}")
    model = SmolVLA.from_pretrained(policy_path)
    model.to(device)
    model.eval()
    
    print(f"Successfully loaded model from: {policy_path}")
    return model

def reset_model(model) -> None:
    if model:
        print("Resetting SmolVLA internal state (action queue).")
        model.reset()

def encode_obs(observation):
    #print(observation)
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
        observation["observation"]["front_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state

def eval(TASK_ENV: Any, model: SmolVLA, observation: Dict) -> None:
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)

    model.update_observation_window(input_rgb_arr, input_state)

    action = model.get_action()  # Get Action according to observation chunk
    #model.reset_observation_window()
    #print(action)
    #print(action.shape)
    TASK_ENV.take_action(action)
