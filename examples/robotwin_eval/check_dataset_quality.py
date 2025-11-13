#!/usr/bin/env python3
"""
检查训练数据集的质量和完整性
"""

import sys
import numpy as np
import torch
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_PATH = "/pfs/pfs-ilWc5D/ziqianwang/lerobot_datasets/name/aloha_agilex_sim/put_bottles_dustbin_v30"

def check_dataset():
    print("=" * 80)
    print("数据集质量检查")
    print("=" * 80)
    
    # 加载数据集
    print("\n加载数据集...")
    ds = LeRobotDataset(
        "name/aloha_agilex_sim/put_bottles_dustbin_v30",
        root=DATASET_PATH
    )
    
    print(f"✓ 数据集加载成功")
    print(f"  总帧数: {len(ds)}")
    print(f"  Episode数: {ds.num_episodes}")
    print(f"  特征: {list(ds.meta.features.keys())}")
    
    # 检查第一帧
    print("\n" + "=" * 80)
    print("检查第一帧数据")
    print("=" * 80)
    frame = ds[0]
    
    # 检查图像
    print("\n图像检查:")
    for cam in ['head_camera', 'left_camera', 'right_camera']:
        key = f'observation.images.{cam}'
        img = frame[key]
        print(f"\n  {cam}:")
        print(f"    Shape: {img.shape}")
        print(f"    Dtype: {img.dtype}")
        print(f"    Min: {img.min():.4f}, Max: {img.max():.4f}, Mean: {img.mean():.4f}")
        print(f"    范围检查: {'✓' if img.min() >= 0 and img.max() <= 1 else '✗ 警告：不在[0,1]范围'}")
        
        # 检查是否全黑或全白
        if img.mean() < 0.01:
            print(f"    ✗ 警告：图像几乎全黑")
        elif img.mean() > 0.99:
            print(f"    ✗ 警告：图像几乎全白")
    
    # 检查state
    print("\n" + "=" * 80)
    print("State检查:")
    print("=" * 80)
    state = frame['observation.state']
    print(f"  Shape: {state.shape}")
    print(f"  Dtype: {state.dtype}")
    print(f"  Values: {state.numpy()}")
    print(f"  Min: {state.min():.4f}, Max: {state.max():.4f}")
    print(f"  非零值数量: {(state != 0).sum()}/{len(state)}")
    
    if (state == 0).all():
        print(f"  ✗ 警告：State全为0")
    
    # 检查action
    print("\n" + "=" * 80)
    print("Action检查:")
    print("=" * 80)
    action = frame['action']
    print(f"  Shape: {action.shape}")
    print(f"  Dtype: {action.dtype}")
    print(f"  Values: {action.numpy()}")
    print(f"  Min: {action.min():.4f}, Max: {action.max():.4f}")
    print(f"  非零值数量: {(action != 0).sum()}/{len(action)}")
    
    if (action == 0).all():
        print(f"  ✗ 警告：Action全为0")
    
    # 检查task
    print("\n" + "=" * 80)
    print("Task检查:")
    print("=" * 80)
    task = frame['task']
    print(f"  Task: {task}")
    print(f"  长度: {len(task)}")
    
    # 统计整个数据集
    print("\n" + "=" * 80)
    print("数据集统计 (采样100帧)")
    print("=" * 80)
    
    sample_indices = np.linspace(0, len(ds)-1, min(100, len(ds)), dtype=int)
    
    states = []
    actions = []
    img_means = []
    
    print("\n采样检查中...")
    for idx in sample_indices:
        frame = ds[int(idx)]
        states.append(frame['observation.state'].numpy())
        actions.append(frame['action'].numpy())
        img_means.append(frame['observation.images.head_camera'].mean().item())
    
    states = np.array(states)
    actions = np.array(actions)
    img_means = np.array(img_means)
    
    print("\nState统计:")
    print(f"  每个维度的均值: {states.mean(axis=0)}")
    print(f"  每个维度的标准差: {states.std(axis=0)}")
    print(f"  非零帧比例: {(states != 0).any(axis=1).mean():.2%}")
    
    print("\nAction统计:")
    print(f"  每个维度的均值: {actions.mean(axis=0)}")
    print(f"  每个维度的标准差: {actions.std(axis=0)}")
    print(f"  非零帧比例: {(actions != 0).any(axis=1).mean():.2%}")
    
    print("\n图像统计:")
    print(f"  平均亮度: {img_means.mean():.4f} ± {img_means.std():.4f}")
    print(f"  最暗帧: {img_means.min():.4f}")
    print(f"  最亮帧: {img_means.max():.4f}")
    
    # 检查action和state的差异
    print("\n" + "=" * 80)
    print("Action vs State差异检查:")
    print("=" * 80)
    
    # 检查连续帧
    print("\n检查前10帧的action和下一帧state的关系:")
    for i in range(min(10, len(ds)-1)):
        frame1 = ds[i]
        frame2 = ds[i+1]
        
        action = frame1['action'].numpy()
        next_state = frame2['observation.state'].numpy()
        curr_state = frame1['observation.state'].numpy()
        
        diff = next_state - curr_state
        action_diff = action - curr_state
        
        print(f"\n  帧 {i} -> {i+1}:")
        print(f"    当前state范围: [{curr_state.min():.3f}, {curr_state.max():.3f}]")
        print(f"    Action范围: [{action.min():.3f}, {action.max():.3f}]")
        print(f"    下一帧state范围: [{next_state.min():.3f}, {next_state.max():.3f}]")
        print(f"    State变化: {np.abs(diff).mean():.4f} (平均)")
        print(f"    Action与当前state的差异: {np.abs(action_diff).mean():.4f} (平均)")
    
    # 检查episode边界
    print("\n" + "=" * 80)
    print("Episode边界检查:")
    print("=" * 80)
    
    print(f"\nEpisode数量: {ds.num_episodes}")
    print(f"前5个episode的长度:")
    for ep_idx in range(min(5, ds.num_episodes)):
        ep_frames = [i for i in range(len(ds)) if ds[i]['episode_index'] == ep_idx]
        print(f"  Episode {ep_idx}: {len(ep_frames)} 帧")
    
    # 检查任务描述的多样性
    print("\n" + "=" * 80)
    print("任务描述多样性检查:")
    print("=" * 80)
    
    tasks = set()
    for idx in sample_indices[:20]:
        frame = ds[int(idx)]
        tasks.add(frame['task'])
    
    print(f"\n采样20帧中的唯一任务描述数: {len(tasks)}")
    print("任务示例:")
    for i, task in enumerate(list(tasks)[:5]):
        print(f"  {i+1}. {task}")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    try:
        check_dataset()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
