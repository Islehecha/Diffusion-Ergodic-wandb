#!/usr/bin/env python3
"""
对 1000*100 规模的 ergodic_dataset 进行统计，计算标准化参数
- 支持按gamma筛选
- 基于训练划分(默认0.9)统计 robot_state 与全轨迹点的均值/方差
- 输出 config_update.yaml 以便自动更新配置
"""

import os
import json
import argparse
import numpy as np
import yaml
from tqdm import tqdm

from diffusion_ergodic.data_process.ergodic_processor import ErgodicDataset
from diffusion_ergodic.main import load_config


def compute_stats(data_dir, trajectory_len, validation_split=0.1, seed=42, gamma_filter=None):
    dataset = ErgodicDataset(data_dir=data_dir, transform=None, max_trajectory_len=trajectory_len)
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(len(indices) * validation_split)
    train_indices = indices[:len(indices)-val_size]

    robot_states = []
    all_points = []
    gammas = []

    for idx in tqdm(train_indices, desc='Collecting stats'):
        sample = dataset[idx]
        if gamma_filter is not None and sample.get('gamma') not in gamma_filter:
            continue
        rs = sample['robot_state']  # shape [4]
        traj = sample['trajectories']  # shape [T, 4]
        robot_states.append(rs)
        all_points.append(traj)
        gammas.append(sample['gamma'])

    robot_states = np.asarray(robot_states, dtype=np.float64)
    all_points = np.concatenate(all_points, axis=0).astype(np.float64)

    stats = {
        'robot_state': {
            'mean': robot_states.mean(axis=0).tolist(),
            'std': robot_states.std(axis=0).tolist(),
            'min': robot_states.min(axis=0).tolist(),
            'max': robot_states.max(axis=0).tolist(),
            'count': int(robot_states.shape[0])
        },
        'trajectory_all_points': {
            'mean': all_points.mean(axis=0).tolist(),
            'std': all_points.std(axis=0).tolist(),
            'min': all_points.min(axis=0).tolist(),
            'max': all_points.max(axis=0).tolist(),
            'count': int(all_points.shape[0])
        },
        'gamma_hist': {str(k): int(v) for k, v in zip(*np.unique(gammas, return_counts=True))}
    }
    return stats


def save_stats(stats, out_dir='dataset_analysis_v2'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'dataset_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    config_update = {
        'normalizer': {
            'robot_state': {
                'mean': stats['robot_state']['mean'],
                'std': stats['robot_state']['std']
            }
        }
    }
    with open(os.path.join(out_dir, 'config_update.yaml'), 'w') as f:
        yaml.safe_dump(config_update, f, sort_keys=False)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Compute normalization stats for ergodic_dataset')
    parser.add_argument('--data_dir', type=str, default='/home/songxy/code/Diffusion-Ergodic/diffusion_ergodic/data/ergodic_dataset_wild_full')
    parser.add_argument('--config', type=str, default='diffusion_ergodic/config/config_ergodic.yaml')
    parser.add_argument('--gamma_filter', type=float, nargs='+', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = compute_stats(args.data_dir, trajectory_len=cfg.trajectory_len, validation_split=cfg.validation_split, seed=cfg.seed, gamma_filter=args.gamma_filter)
    out_dir = save_stats(stats)

    print('Done. Stats saved to', out_dir)
    print('Suggested config update:')
    print('normalizer:')
    print('  robot_state:')
    print('    mean:', stats['robot_state']['mean'])
    print('    std:', stats['robot_state']['std'])


if __name__ == '__main__':
    main()

