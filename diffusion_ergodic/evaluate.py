import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion_ergodic.models.diffusion_ergodic import ErgodicDiffusionModel
from diffusion_ergodic.data_process.ergodic_processor import ErgodicDataset, ErgodicTransform
from diffusion_ergodic.main import load_config

# 轻量后处理：边界裁剪 + 简单滑动平均，避免生成轨迹越界与过直
WORKSPACE_BOUNDS = np.array([[0.0, 3.5], [-1.0, 3.5]])

def _clamp_and_smooth(traj: np.ndarray, window: int = 3) -> np.ndarray:
    t = traj.copy()
    # 仅处理 x,y
    t[:, 0] = np.clip(t[:, 0], WORKSPACE_BOUNDS[0, 0], WORKSPACE_BOUNDS[0, 1])
    t[:, 1] = np.clip(t[:, 1], WORKSPACE_BOUNDS[1, 0], WORKSPACE_BOUNDS[1, 1])
    if window and window > 1 and len(t) >= window:
        k = np.ones(window) / window
        # 保存首点，避免平滑破坏起点
        x0, y0 = t[0, 0], t[0, 1]
        t[:, 0] = np.convolve(t[:, 0], k, mode='same')
        t[:, 1] = np.convolve(t[:, 1], k, mode='same')
        t[0, 0], t[0, 1] = x0, y0
    return t


def compute_coverage_metric(trajectory, distribution, grid_size=(32, 32)):
    """
    计算轨迹对分布的覆盖度指标（简化版）
    
    Args:
        trajectory: [T, 2] - 轨迹点
        distribution: [H, W] - 分布网格
        grid_size: 网格大小
    
    Returns:
        coverage_score: 覆盖度分数 [0, 1]
    """
    # 统一工作空间边界（与数据生成一致）
    x_min, x_max = 0.0, 3.5
    y_min, y_max = -1.0, 3.5

    # 创建轨迹占据网格
    traj_grid = np.zeros(grid_size)
    
    # 映射轨迹点到网格
    for point in trajectory:
        x, y = point[:2]
        # 将坐标按边界归一化并映射到网格坐标
        if np.isnan(x) or np.isnan(y):
            continue
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        x_norm = np.clip(x_norm, 0.0, 0.999)
        y_norm = np.clip(y_norm, 0.0, 0.999)

        grid_x = int(x_norm * grid_size[0])
        grid_y = int(y_norm * grid_size[1])
        
        # 标记网格为已访问
        traj_grid[grid_y, grid_x] = 1
    
    # 计算轨迹占据的重要区域（按分布权重）
    importance_covered = np.sum(traj_grid * distribution)
    total_importance = np.sum(distribution)
    
    # 计算覆盖率
    if total_importance > 0:
        coverage_score = importance_covered / total_importance
    else:
        coverage_score = 0
    
    return coverage_score


def compute_smoothness_metric(trajectory):
    """
    计算轨迹平滑度指标
    
    Args:
        trajectory: [T, state_dim] - 轨迹状态
    
    Returns:
        smoothness_score: 平滑度分数 [0, 1]，值越高越平滑
    """
    if len(trajectory) < 3:
        return 1.0  # 点太少无法评估平滑度
    
    # 只考虑位置
    positions = trajectory[:, :2]
    
    # 计算相邻点之间的角度变化
    vectors = np.diff(positions, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.abs(np.diff(angles))
    
    # 处理角度环绕（>π的差值）
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
    
    # 计算平均角度变化
    mean_angle_diff = np.mean(angle_diffs)
    
    # 将平均角度变化映射到[0, 1]范围，0表示变化最大（π），1表示没有变化
    smoothness_score = 1.0 - mean_angle_diff / np.pi
    
    return smoothness_score


def evaluate_model(model, dataset, device, num_samples=None):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        device: 运算设备
        num_samples: 要评估的样本数量，None表示全部
    
    Returns:
        metrics: 包含各项指标的字典
    """
    model.eval()
    
    metrics = {
        'coverage': [],
        'smoothness': [],
        'mse': []
    }
    
    # 限制评估样本数量
    indices = list(range(len(dataset)))
    if num_samples is not None and num_samples < len(indices):
        indices = indices[:num_samples]
    
    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        
        # 准备输入
        distribution = sample['distribution'].unsqueeze(0).to(device)
        robot_state = sample['robot_state'].unsqueeze(0).to(device)
        gt_trajectory = sample['trajectories'].numpy()

        # 仅截掉尾部零填充，避免中间断段被直线连接
        gt_nonzero = ~np.all(gt_trajectory == 0, axis=1)
        gt_valid_len = np.where(gt_nonzero)[0].max() + 1 if gt_nonzero.any() else len(gt_trajectory)
        gt_valid = gt_trajectory[:gt_valid_len]

        # 模型推理
        with torch.no_grad():
            outputs = model.inference({
                'distribution': distribution,
                'robot_state': robot_state
            })
        
        # 获取预测轨迹
        pred_trajectory = outputs['prediction'][0].cpu().numpy()
        # 轻量后处理：边界裁剪 + 平滑
        pred_trajectory = _clamp_and_smooth(pred_trajectory, window=3)
        # 仅截掉尾部零填充
        pred_nonzero = ~np.all(pred_trajectory == 0, axis=1)
        pred_valid_len = np.where(pred_nonzero)[0].max() + 1 if pred_nonzero.any() else len(pred_trajectory)
        pred_valid = pred_trajectory[:pred_valid_len]

        # 计算指标
        # 1. 覆盖度
        coverage = compute_coverage_metric(
            pred_valid,
            sample['distribution'][0].numpy(), 
            grid_size=(32, 32)
        )
        metrics['coverage'].append(coverage)
        
        # 2. 平滑度
        smoothness = compute_smoothness_metric(pred_valid)
        metrics['smoothness'].append(smoothness)
        
        # 3. 与groundtruth的MSE
        # 只计算有效长度的MSE
        valid_len = min(len(gt_valid), len(pred_valid))
        mse = np.mean((gt_valid[:valid_len, :2] - pred_valid[:valid_len, :2])**2)
        metrics['mse'].append(mse)
    
    # 计算平均值
    for key in metrics:
        metrics[key] = {
            'mean': np.mean(metrics[key]),
            'std': np.std(metrics[key]),
            'min': np.min(metrics[key]),
            'max': np.max(metrics[key])
        }
    
    return metrics


def visualize_results(model, dataset, device, save_dir, num_samples=5):
    """
    可视化模型生成的轨迹与地面真相对比
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        device: 运算设备
        save_dir: 保存结果的目录
        num_samples: 要可视化的样本数量
    """
    model.eval()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择样本
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # 准备输入
        distribution = sample['distribution'].unsqueeze(0).to(device)
        robot_state = sample['robot_state'].unsqueeze(0).to(device)
        gt_trajectory = sample['trajectories'].numpy()

        # 仅截掉尾部零填充
        gt_nonzero = ~np.all(gt_trajectory == 0, axis=1)
        gt_valid_len = np.where(gt_nonzero)[0].max() + 1 if gt_nonzero.any() else len(gt_trajectory)
        gt_valid = gt_trajectory[:gt_valid_len]

        # 模型推理
        with torch.no_grad():
            outputs = model.inference({
                'distribution': distribution,
                'robot_state': robot_state
            })
        
        # 获取预测轨迹
        pred_trajectory = outputs['prediction'][0].cpu().numpy()
        # 轻量后处理：边界裁剪 + 平滑
        pred_trajectory = _clamp_and_smooth(pred_trajectory, window=3)
        pred_nonzero = ~np.all(pred_trajectory == 0, axis=1)
        pred_valid_len = np.where(pred_nonzero)[0].max() + 1 if pred_nonzero.any() else len(pred_trajectory)
        pred_valid = pred_trajectory[:pred_valid_len]

        # 计算指标
        coverage = compute_coverage_metric(
            pred_valid, 
            sample['distribution'][0].numpy()
        )
        smoothness = compute_smoothness_metric(pred_valid)
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制分布
        distribution_np = sample['distribution'][0].numpy()
        # 使用与数据一致的工作空间边界
        im = ax.imshow(
            distribution_np,
            cmap='viridis',
            alpha=0.7,
            origin='lower',
            extent=[0.0, 3.5, -1.0, 3.5]
        )
        fig.colorbar(im, ax=ax, label='Distribution Density')
        
        # 绘制地面真相轨迹
        ax.plot(gt_valid[:, 0], gt_valid[:, 1], 'g-', linewidth=2, label='Ground Truth')
        
        # 绘制生成的轨迹
        ax.plot(pred_valid[:, 0], pred_valid[:, 1], 'r--', linewidth=2, label='Generated')
        
        # 标记起点
        ax.scatter(gt_valid[0, 0], gt_valid[0, 1], c='b', s=100, marker='o', label='Start')
        
        # 添加标题和标签
        ax.set_title(f'Sample {i} - Coverage: {coverage:.3f}, Smoothness: {smoothness:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='diffusion_ergodic/config/config_ergodic.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=50, help='要评估的样本数量')
    parser.add_argument('--vis_samples', type=int, default=10, help='要可视化的样本数量')
    parser.add_argument('--output_dir', type=str, default='/mnt/sfs_turbo/songxy/results/evaluation', help='输出目录')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ErgodicDiffusionModel(config).to(device)
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f"加载检查点 '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"检查点已加载。")
    else:
        print(f"未找到检查点 '{args.checkpoint}'")
        return
    
    # 创建数据集
    transform = ErgodicTransform(config)
    dataset = ErgodicDataset(
        data_dir=config.data.data_dir,
        transform=transform,
        max_trajectory_len=config.data.trajectory_len
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(model, dataset, device, args.num_samples)
    
    # 打印和保存评估结果
    print("\n评估结果:")
    for metric_name, metric_values in metrics.items():
        print(f"{metric_name}:")
        for stat_name, stat_value in metric_values.items():
            print(f"  {stat_name}: {stat_value:.6f}")
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    # 可视化结果
    print("\n生成可视化结果...")
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    visualize_results(model, dataset, device, vis_dir, args.vis_samples)
    
    print(f"评估完成。结果保存在 {args.output_dir}")


if __name__ == '__main__':
    main()
