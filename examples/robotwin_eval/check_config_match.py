#!/usr/bin/env python3
"""
检查eval配置与训练配置的匹配情况
"""

import json
import sys
from pathlib import Path

# 配置路径
CHECKPOINT_PATH = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/050000/pretrained_model"

def load_config():
    config_file = Path(CHECKPOINT_PATH) / "config.json"
    with open(config_file) as f:
        return json.load(f)

def load_preprocessor_config():
    config_file = Path(CHECKPOINT_PATH) / "policy_preprocessor.json"
    with open(config_file) as f:
        return json.load(f)

def main():
    print("=" * 80)
    print("配置匹配检查")
    print("=" * 80)
    
    config = load_config()
    preprocessor_config = load_preprocessor_config()
    
    print("\n模型配置:")
    print(f"  ✓ n_action_steps: {config['n_action_steps']}")
    print(f"  ✓ chunk_size: {config['chunk_size']}")
    print(f"  ✓ n_obs_steps: {config['n_obs_steps']}")
    print(f"  ✓ tokenizer_max_length: {config['tokenizer_max_length']}")
    print(f"  ✓ freeze_vision_encoder: {config['freeze_vision_encoder']}")
    print(f"  ✓ adapt_to_pi_aloha: {config['adapt_to_pi_aloha']}")
    print(f"  ✓ use_delta_joint_actions_aloha: {config['use_delta_joint_actions_aloha']}")
    print(f"  ✓ resize_imgs_with_padding: {config['resize_imgs_with_padding']}")
    
    print("\n输入特征:")
    for key, feat in config['input_features'].items():
        print(f"  ✓ {key}: type={feat['type']}, shape={feat['shape']}")
    
    print("\n输出特征:")
    for key, feat in config['output_features'].items():
        print(f"  ✓ {key}: type={feat['type']}, shape={feat['shape']}")
    
    print("\nNormalization配置:")
    print(f"  ✓ normalization_mapping: {config['normalization_mapping']}")
    
    print("\nPreprocessor步骤:")
    for i, step in enumerate(preprocessor_config['steps']):
        print(f"  {i}. {step['registry_name']}")
        if step['registry_name'] == 'tokenizer_processor':
            print(f"     - max_length: {step['config']['max_length']}")
            print(f"     - padding: {step['config']['padding']}")
        elif step['registry_name'] == 'normalizer_processor':
            print(f"     - norm_map: {step['config']['norm_map']}")
            if 'state_file' in step:
                print(f"     - state_file: {step['state_file']}")
    
    print("\n" + "=" * 80)
    print("建议的eval配置:")
    print("=" * 80)
    print(f"n_action_steps: {config['n_action_steps']}")
    print(f"tokenizer_max_length: {config['tokenizer_max_length']}")
    print("相机顺序: head_camera, left_camera, right_camera")
    print("图像格式: CHW, float32, [0, 1]")
    print("图像shape: [3, 480, 640]")
    print("==========")

if __name__ == "__main__":
    main()
