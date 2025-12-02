# test_classifier_guidance.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

# 添加项目根目录到路径
sys.path.append('../../../')

# 正确的导入路径
from diffusion_ergodic.models.diffusion_ergodic import ErgodicDiffusionModel
from diffusion_ergodic.data_process.ergodic_processor import get_data_loaders

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 转换为对象以便属性访问
    class Config:
        def __init__(self, dic):
            for key, value in dic.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
        
        def __repr__(self):
            attrs = ', '.join([f"{k}={v}" for k, v in self.__dict__.items()])
            return f"Config({attrs})"
    
    config = Config(config_dict)
    
    # 为了兼容模型代码，添加一些顶层属性
    # 扩散参数
    if hasattr(config, 'diffusion'):
        config.beta_min = config.diffusion.beta_min
        config.beta_max = config.diffusion.beta_max
        config.diffusion_model_type = config.diffusion.model_type
        config.diffusion_steps = config.diffusion.steps
        
        # 添加对混合训练模式的支持
        if hasattr(config.diffusion, 'training_mode'):
            config.training_mode = config.diffusion.training_mode
        if hasattr(config.diffusion, 'mix_alpha'):
            config.mix_alpha = config.diffusion.mix_alpha
        # 添加起点约束权重
        if hasattr(config.diffusion, 'start_point_coef'):
            config.start_point_coef = config.diffusion.start_point_coef
    
    # 数据参数
    if hasattr(config, 'data'):
        config.data_dir = config.data.data_dir
        config.trajectory_len = config.data.trajectory_len
        config.robot_state_dim = config.data.robot_state_dim
        config.distribution_dim = config.data.distribution_dim
        config.validation_split = config.data.validation_split
        config.shuffle_dataset = config.data.shuffle_dataset
        config.num_workers = config.data.num_workers
        config.seed = config.data.seed
    
    # 模型参数
    if hasattr(config, 'model'):
        config.hidden_dim = config.model.hidden_dim
        config.encoder_depth = config.model.encoder_depth
        config.decoder_depth = config.model.decoder_depth
        config.num_heads = config.model.num_heads
        config.encoder_drop_path_rate = config.model.encoder_drop_path_rate
        config.decoder_drop_path_rate = config.model.decoder_drop_path_rate
    
    # 训练参数
    if hasattr(config, 'training'):
        config.batch_size = config.training.batch_size
        config.learning_rate = config.training.learning_rate
        config.weight_decay = config.training.weight_decay
        config.num_epochs = config.training.num_epochs
        config.device = config.training.device
        config.output_dir = config.training.output_dir
    
    # 处理需要特殊处理的配置项
    if hasattr(config, 'normalizer') and hasattr(config.normalizer, 'robot_state'):
        config.normalizer.robot_state.mean = torch.tensor(config.normalizer.robot_state.mean)
        config.normalizer.robot_state.std = torch.tensor(config.normalizer.robot_state.std)
    
    return config

def visualize_guidance_effect(model, data_batch, device, save_dir='results/guidance_test'):
    """可视化引导前后的轨迹对比"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备输入数据
    inputs = {
        'distribution': data_batch['distribution'].to(device),
        'robot_state': data_batch['trajectories'][:, 0, :].to(device)
    }
    
    # 设置不同的引导强度
    guidance_scales = [0.0, 1.0, 2.0, 3.0, 5.0]
    
    for i, scale in enumerate(guidance_scales):
        # 设置当前引导强度
        model.guidance_scale = scale
        
        # 推理
        with torch.no_grad():
            outputs = model.inference(inputs)
        
        # 提取轨迹和相关数据
        pred_traj = outputs['trajectories'].cpu().numpy()
        gt_traj = data_batch['trajectories'].cpu().numpy()
        
        # 获取原始轨迹（如果有）
        if 'original_trajectories' in outputs:
            orig_traj = outputs['original_trajectories'].cpu().numpy()
        else:
            orig_traj = None
        
        # 获取分布和高斯中心
        distribution = data_batch['distribution'].cpu().numpy()
        if 'gaussian_centers' in data_batch:
            centers = data_batch['gaussian_centers'].cpu().numpy()
        else:
            # 如果数据中没有高斯中心，则尝试从分布中估计
            centers = estimate_distribution_centers(distribution)
        
        # 为每个批次样本创建图像
        for b in range(min(3, pred_traj.shape[0])):
            plt.figure(figsize=(10, 8))
            
            # 显示分布
            dist = distribution[b, 0]
            plt.imshow(dist, origin='lower', extent=[0, 1, 0, 1], 
                       cmap='viridis', alpha=0.7)
            
            # 绘制中心点
            if centers is not None:
                for c_idx, center in enumerate(centers[b]):
                    plt.scatter(center[0], center[1], c='red', s=100, 
                                marker='o', label=f'Center {c_idx+1}' if c_idx == 0 else None)
            
            # 绘制原始轨迹（如果有）
            if orig_traj is not None:
                plt.plot(orig_traj[b, :, 0], orig_traj[b, :, 1], 'y--', 
                         label='Original', linewidth=2)
            
            # 绘制引导后轨迹
            plt.plot(pred_traj[b, :, 0], pred_traj[b, :, 1], 'r-', 
                     label='Guided', linewidth=3)
            
            # 绘制真实轨迹
            plt.plot(gt_traj[b, :, 0], gt_traj[b, :, 1], 'b-', 
                     label='Ground Truth', linewidth=2)
            
            # 标记起点和终点
            plt.scatter(pred_traj[b, 0, 0], pred_traj[b, 0, 1], c='lime', s=150, 
                        marker='o', label='Start')
            plt.scatter(pred_traj[b, -1, 0], pred_traj[b, -1, 1], c='cyan', s=150, 
                        marker='*', label='End (Guided)')
            
            # 添加覆盖分数信息（如果有）
            if 'guided_coverage_score' in outputs:
                orig_score = outputs['original_coverage_score'][b].item()
                guided_score = outputs['guided_coverage_score'][b].item()
                plt.title(f'Guidance Scale: {scale} - Coverage Score: {orig_score:.3f} → {guided_score:.3f}')
            else:
                plt.title(f'Guidance Scale: {scale}')
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            save_path = os.path.join(save_dir, f'sample_{b}_guidance_{scale}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"保存图像: {save_path}")

def estimate_distribution_centers(distribution, threshold=0.7):
    """从分布图中估计高斯中心位置"""
    B, C, H, W = distribution.shape
    centers = []
    
    for b in range(B):
        dist = distribution[b, 0]
        # 归一化
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        
        # 找到局部最大值
        batch_centers = []
        for _ in range(5):  # 最多找5个中心
            if dist.max() < threshold:
                break
                
            # 找最大值位置
            idx = np.unravel_index(dist.argmax(), dist.shape)
            y, x = idx
            
            # 转换为归一化坐标
            center_x = x / (W - 1)
            center_y = y / (H - 1)
            batch_centers.append([center_x, center_y])
            
            # 将已找到的最大值周围置零，避免重复
            y_min, y_max = max(0, y-3), min(H, y+4)
            x_min, x_max = max(0, x-3), min(W, x+4)
            dist[y_min:y_max, x_min:x_max] = 0
        
        centers.append(batch_centers)
    
    # 填充到相同长度
    max_centers = max(len(c) for c in centers)
    padded_centers = []
    for c in centers:
        while len(c) < max_centers:
            c.append([0, 0])  # 用[0,0]填充
        padded_centers.append(c)
    
    return np.array(padded_centers)

def calculate_metrics(trajectory, distribution, centers=None):
    """计算轨迹的平滑度和覆盖率指标"""
    # 平滑度指标：基于轨迹段方向变化
    segments = trajectory[1:] - trajectory[:-1]
    angles = np.arctan2(segments[1:, 1], segments[1:, 0]) - np.arctan2(segments[:-1, 1], segments[:-1, 0])
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi  # 归一化到[-pi, pi]
    smoothness = 1.0 - (np.abs(angles).mean() / np.pi)  # 1.0是最平滑
    
    # 覆盖率指标
    coverage = 0.0
    if centers is not None:
        # 到最近中心的最小距离
        min_distances = []
        for point in trajectory:
            dists = np.linalg.norm(centers - point[:2], axis=1)
            min_distances.append(np.min(dists))
        
        # 平均最小距离越小，覆盖率越高
        avg_min_dist = np.mean(min_distances)
        coverage = np.exp(-5.0 * avg_min_dist)  # 缩放因子5.0可以调整
    
    return {
        'smoothness': smoothness,
        'coverage': coverage
    }

def main():
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    model_path = 'diffusion_ergodic/trained/best_model.pth'
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    _, val_loader = get_data_loaders(config)
    
    # 创建模型
    model = ErgodicDiffusionModel(config).to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取一个批次的数据
    data_batch = next(iter(val_loader))
    
    # 测试不同引导强度
    visualize_guidance_effect(model, data_batch, device)
    
    print("测试完成!")

if __name__ == '__main__':
    main()