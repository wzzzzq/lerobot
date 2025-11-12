#!/bin/bash

# 转换 put_bottles_dustbin 数据集到 LeRobot 格式
# 数据路径: /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/put_bottles_dustbin-demo_clean-700

python RoboTwin/script/robotwin2lerobot.py \
    --raw-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/put_bottles_dustbin-demo_clean-700 \
    --repo-id name/aloha_agilex_sim \
    --instruction-dir /pfs/pfs-ilWc5D/VLA-MoE/Eval-RoboTwin/RoboTwin/new-clean-data/put_bottles_dustbin-demo_clean-700 \
    --task "put_bottles_dustbin"
