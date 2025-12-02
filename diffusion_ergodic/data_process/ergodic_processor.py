import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ErgodicDataset(Dataset):
    def __init__(self, data_dir='diffusion_ergodic/data/ergodic_dataset', 
                 transform=None, max_trajectory_len=100, use_index=True):
        """
        Ergodic 数据集加载器
        
        Args:
            data_dir: 数据目录
            transform: 可选的数据转换
            max_trajectory_len: 最大轨迹长度（用于填充）
            use_index: 是否使用索引文件加速加载
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_trajectory_len = max_trajectory_len
        
        self.distributions_dir = os.path.join(data_dir, 'distributions')
        self.trajectories_dir = os.path.join(data_dir, 'trajectories')
        
        # 初始化数据对列表
        self.data_pairs = []
        
        # 检查索引文件
        index_path = os.path.join(data_dir, 'dataset_index.json')
        if use_index and os.path.exists(index_path):
            # 从索引文件加载
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                # 处理已有的索引文件格式
                if isinstance(index_data, dict) and 'distributions' in index_data and 'trajectories' in index_data:
                    distributions = index_data['distributions']
                    trajectories = index_data['trajectories']
                    
                    # 构建分布ID到文件名的映射
                    dist_id_to_file = {}
                    for dist_id in distributions:
                        dist_id_to_file[dist_id] = f"{dist_id}.json"
                    
                    # 构建配对关系
                    for traj_name in trajectories:
                        # 从轨迹名中提取分布ID
                        # 格式例如: "dist_0000_traj_0000_g0.001000"
                        parts = traj_name.split('_')
                        if len(parts) >= 2:
                            dist_id = f"{parts[0]}_{parts[1]}"
                            if dist_id in dist_id_to_file:
                                self.data_pairs.append({
                                    'distribution_file': dist_id_to_file[dist_id],
                                    'trajectory_file': f"{traj_name}.json"
                                })
                    
                    print(f"从索引加载了 {len(self.data_pairs)} 对分布-轨迹数据")
                else:
                    # 不是预期的格式，重建索引
                    print("索引文件格式不符合预期，重建索引...")
                    self._build_index(index_path)
            except Exception as e:
                print(f"读取索引文件出错: {e}，重建索引...")
                self._build_index(index_path)
        else:
            # 手动构建索引
            self._build_index(index_path)
    
    def _build_index(self, index_path):
        """构建数据集索引"""
        print("构建数据集索引...")
        
        # 加载所有分布文件
        dist_files = [f for f in os.listdir(self.distributions_dir) if f.endswith('.json')]
        
        # 构建分布ID到文件名的映射
        dist_id_to_file = {}
        for dist_file in dist_files:
            try:
                with open(os.path.join(self.distributions_dir, dist_file), 'r') as f:
                    dist_data = json.load(f)
                    dist_id = dist_data['id']
                    dist_id_to_file[dist_id] = dist_file
            except Exception as e:
                print(f"读取分布文件 {dist_file} 出错: {e}")
        
        # 查找匹配的轨迹
        for traj_file in os.listdir(self.trajectories_dir):
            if traj_file.endswith('.json'):
                try:
                    with open(os.path.join(self.trajectories_dir, traj_file), 'r') as f:
                        traj_data = json.load(f)
                        dist_id = traj_data['distribution_id']
                        if dist_id in dist_id_to_file:
                            self.data_pairs.append({
                                'distribution_file': dist_id_to_file[dist_id],
                                'trajectory_file': traj_file
                            })
                except Exception as e:
                    print(f"读取轨迹文件 {traj_file} 出错: {e}")
        
        # 保存索引文件（使用新的格式，直接保存配对关系）
        try:
            with open(index_path, 'w') as f:
                json.dump(self.data_pairs, f)
            
            print(f"构建并保存了 {len(self.data_pairs)} 对分布-轨迹数据的索引")
        except Exception as e:
            print(f"保存索引文件出错: {e}")

    def __len__(self):
        return len(self.data_pairs)
    
    def _generate_distribution_grid(self, dist_data, grid_size=(32, 32)):
        """将高斯分布参数转换为网格化分布"""
        params = dist_data['params']
        n_gaussians = params['n_gaussians']
        centers = np.array(params['centers'])
        covs = np.array(params['covs'])
        weights = np.array(params['weights'])
        
        # 获取工作空间边界
        bounds = np.array(dist_data['workspace_bounds'])
        x_min, y_min = bounds[0]
        x_max, y_max = bounds[1]
        
        # 创建网格
        x = np.linspace(x_min, x_max, grid_size[0])
        y = np.linspace(y_min, y_max, grid_size[1])
        xx, yy = np.meshgrid(x, y)
        pos = np.dstack((xx, yy))
        
        # 计算高斯混合分布
        grid = np.zeros(grid_size)
        
        for i in range(n_gaussians):
            center = centers[i]
            if isinstance(covs[i], (int, float)):
                cov = np.diag([covs[i], covs[i]])
            else:
                # 确保是2x2矩阵或者长度为2的向量
                if len(covs[i]) == 2:
                    cov = np.diag(covs[i])
                else:
                    cov = np.array(covs[i]).reshape(2, 2)
                    
            weight = weights[i]
            
            # 计算多元高斯分布
            inv_cov = np.linalg.inv(cov)
            diff = pos - center
            mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=2)
            gaussian = np.exp(-0.5 * mahalanobis) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
            grid += weight * gaussian
        
        # 归一化
        if grid.max() > 0:
            grid = grid / grid.max()
            
        return grid
    
    def _process_trajectory(self, traj_data):
        """处理轨迹数据"""
        states = np.array(traj_data['states'])
        
        # 提取位置和角度
        positions = states[:, :2]  # 前两列是位置
        headings = states[:, 2:3] if states.shape[1] > 2 else np.zeros((states.shape[0], 1))
        
        # 计算速度（如果没有提供）
        if states.shape[1] <= 3:  # 只有位置和可能的角度，没有速度
            velocities = np.zeros((states.shape[0], 1))
            if states.shape[0] > 1:
                # 从位置差计算速度
                time_step = traj_data['time_step']
                pos_diff = np.diff(positions, axis=0)
                velocities = np.vstack([np.zeros((1, 1)), np.sqrt(np.sum(pos_diff**2, axis=1))[:, None] / time_step])
        else:
            velocities = states[:, 3:4]  # 假设第四列是速度
        
        # 组合成标准机器人状态 [x, y, theta, v]
        robot_states = np.hstack([positions, headings, velocities])
        
        # 获取控制输入
        controls = np.array(traj_data['controls']) if 'controls' in traj_data else np.zeros((robot_states.shape[0], 2))
        
        # 裁剪或填充到最大长度
        if len(robot_states) > self.max_trajectory_len:
            # 均匀采样
            indices = np.linspace(0, len(robot_states)-1, self.max_trajectory_len).astype(int)
            robot_states = robot_states[indices]
            controls = controls[indices] if len(controls) > 0 else controls
        elif len(robot_states) < self.max_trajectory_len:
            # 填充
            pad_length = self.max_trajectory_len - len(robot_states)
            robot_states = np.vstack([robot_states, np.zeros((pad_length, robot_states.shape[1]))])
            if len(controls) > 0:
                controls = np.vstack([controls, np.zeros((pad_length, controls.shape[1]))])
        
        return {
            'states': robot_states,
            'controls': controls,
            'time_step': traj_data['time_step'],
            'total_time': traj_data['total_time'],
            'ergodic_metric': traj_data['ergodic_metric'],
            'gamma': traj_data['gamma']
        }
    
    def __getitem__(self, idx):
        if idx >= len(self.data_pairs):
            raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.data_pairs)-1})")
        
        pair = self.data_pairs[idx]
        
        # 加载分布数据
        dist_path = os.path.join(self.distributions_dir, pair['distribution_file'])
        try:
            with open(dist_path, 'r') as f:
                dist_data = json.load(f)
        except Exception as e:
            print(f"读取分布文件 {dist_path} 出错: {e}")
            # 返回一个空的样本
            return self._create_empty_sample()
        
        # 加载轨迹数据
        traj_path = os.path.join(self.trajectories_dir, pair['trajectory_file'])

        try:
            with open(traj_path, 'r') as f:
                traj_data = json.load(f)
        except Exception as e:
            print(f"读取轨迹文件 {traj_path} 出错: {e}")
            # 返回一个空的样本
            return self._create_empty_sample()
        
        # 生成分布网格
        try:
            distribution_grid = self._generate_distribution_grid(dist_data)
        except Exception as e:
            print(f"生成分布网格出错: {e}")
            distribution_grid = np.zeros((32, 32))
        
        # 处理轨迹
        try:
            trajectory_data = self._process_trajectory(traj_data)
        except Exception as e:
            print(f"处理轨迹数据出错: {e}")
            # 返回一个空的样本
            return self._create_empty_sample()
        
        # 构建样本，确保gaussian_params的centers等数组有固定大小
        max_gaussians = 10  # 根据你的数据集最大高斯数量调整
        
        # 准备高斯参数
        n_gaussians = dist_data['params']['n_gaussians']
        centers = np.array(dist_data['params']['centers'])
        covs = np.array(dist_data['params']['covs'])
        weights = np.array(dist_data['params']['weights'])
        
        # 填充到固定大小
        padded_centers = np.zeros((max_gaussians, 2))
        padded_covs = np.zeros(max_gaussians)
        padded_weights = np.zeros(max_gaussians)
        
        # 复制实际数据
        padded_centers[:n_gaussians] = centers[:n_gaussians]
        padded_covs[:n_gaussians] = covs[:n_gaussians]
        padded_weights[:n_gaussians] = weights[:n_gaussians]
        
        sample = {
            'distribution_id': dist_data['id'],
            'trajectory_id': os.path.splitext(pair['trajectory_file'])[0],
            'distribution': distribution_grid,
            'robot_state': trajectory_data['states'][0],  # 初始状态
            'trajectories': trajectory_data['states'],
            'controls': trajectory_data['controls'],
            'time_step': trajectory_data['time_step'],
            'total_time': trajectory_data['total_time'],
            'ergodic_metric': trajectory_data['ergodic_metric'],
            'gamma': trajectory_data['gamma'],
            'gaussian_params': {
                'n_gaussians': n_gaussians,
                'centers': padded_centers,
                'covs': padded_covs,
                'weights': padded_weights
            }
        }
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        # 转换为张量
        sample['distribution'] = torch.FloatTensor(sample['distribution']).unsqueeze(0)  # 添加通道维度
        sample['robot_state'] = torch.FloatTensor(sample['robot_state'])
        sample['trajectories'] = torch.FloatTensor(sample['trajectories'])
        sample['controls'] = torch.FloatTensor(sample['controls'])
        sample['gaussian_params']['centers'] = torch.FloatTensor(sample['gaussian_params']['centers'])
        sample['gaussian_params']['covs'] = torch.FloatTensor(sample['gaussian_params']['covs'])
        sample['gaussian_params']['weights'] = torch.FloatTensor(sample['gaussian_params']['weights'])
        
        return sample
    
    def _create_empty_sample(self):
        """创建一个空的样本，用于错误处理；返回与正常样本完全一致的张量类型与形状"""
        max_gaussians = 10  # 与 __getitem__ 中的固定大小保持一致
        return {
            'distribution_id': 'error',
            'trajectory_id': 'error',
            'distribution': torch.zeros((1, 32, 32)),
            'robot_state': torch.zeros(4),
            'trajectories': torch.zeros((self.max_trajectory_len, 4)),
            'controls': torch.zeros((self.max_trajectory_len, 2)),
            'time_step': 0.1,
            'total_time': 0.0,
            'ergodic_metric': 0.0,
            'gamma': 0.0,
            'gaussian_params': {
                'n_gaussians': 0,
                'centers': torch.zeros((max_gaussians, 2)),
                'covs': torch.zeros(max_gaussians),
                'weights': torch.zeros(max_gaussians)
            }
        }

    def visualize_sample(self, idx, save_path=None):
        """可视化一个样本"""
        if idx >= len(self.data_pairs):
            print(f"索引 {idx} 超出范围 (0-{len(self.data_pairs)-1})")
            return
        
        sample = self[idx]
        
        # 转换回 numpy 以便绘图
        distribution = sample['distribution'][0].numpy()
        trajectories = sample['trajectories'].numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制分布
        im = ax.imshow(distribution, cmap='viridis', alpha=0.7, origin='lower')
        fig.colorbar(im, ax=ax, label='分布密度')
        
        # 绘制轨迹
        # 过滤掉填充的零值
        mask = ~np.all(trajectories[:, :2] == 0, axis=1)
        valid_traj = trajectories[mask]
        
        if len(valid_traj) > 0:
            ax.plot(valid_traj[:, 0], valid_traj[:, 1], 'r-', linewidth=2, label='轨迹')
            ax.scatter(valid_traj[0, 0], valid_traj[0, 1], c='g', s=100, marker='o', label='起点')
        
        # 添加标题和标签
        ax.set_title(f'分布 ID: {sample["distribution_id"]}, 轨迹 ID: {sample["trajectory_id"]}')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.legend()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class ErgodicTransform:
    """数据预处理和归一化，注意trajectories 没有被归一化"""
    def __init__(self, config):
        self.config = config
        
        # 初始化归一化参数
        self.robot_state_mean = config.normalizer.robot_state.mean
        self.robot_state_std = config.normalizer.robot_state.std
    
    def __call__(self, sample):
        """
        应用变换到样本
        """
        # 将 robot_state 转换为张量（如果它是 numpy 数组）
        if isinstance(sample['robot_state'], np.ndarray):
            sample['robot_state'] = torch.tensor(sample['robot_state'], dtype=torch.float32)
        
        # 确保均值和标准差也是张量
        if isinstance(self.robot_state_mean, np.ndarray):
            self.robot_state_mean = torch.tensor(self.robot_state_mean, dtype=torch.float32)
        if isinstance(self.robot_state_std, np.ndarray):
            self.robot_state_std = torch.tensor(self.robot_state_std, dtype=torch.float32)
        
        # 现在它们都是张量，可以安全地执行减法操作
        sample['robot_state'] = (sample['robot_state'] - self.robot_state_mean) / self.robot_state_std
        
        # 只将轨迹转换为张量，不进行标准化
        if isinstance(sample['trajectories'], np.ndarray):
            sample['trajectories'] = torch.tensor(sample['trajectories'], dtype=torch.float32)
        
        return sample

def get_data_loaders(config):
    """创建数据加载器"""
    transform = ErgodicTransform(config)
    
    # 创建数据集
    dataset = ErgodicDataset(
        data_dir=config.data_dir,
        transform=transform,
        max_trajectory_len=config.trajectory_len
    )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.validation_split * dataset_size))
    
    if config.shuffle_dataset:
        np.random.seed(config.seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def reconstruct_distribution(centers, covs, weights, workspace_bounds, grid_size=50):
    """
    基于高斯混合模型参数重建2D分布网格，将抽象的高斯混合模型参数转换为固定尺寸、网格化（类似图像）的概率密度图
    这种格式是神经网络（尤其是处理空间信息的模型）能够高效理解和处理的输入。
    
    参数:
    - centers: 高斯分布中心点列表
    - covs: 协方差参数列表
    - weights: 权重列表
    - workspace_bounds: 工作空间边界 [[x_min, x_max], [y_min, y_max]]
    - grid_size: 网格尺寸
    
    返回:
    - 2D网格形式的分布
    """
    x_min, x_max = workspace_bounds[0]
    y_min, y_max = workspace_bounds[1]
    
    # 创建网格
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 初始化结果网格
    Z = np.zeros_like(X)
    
    # 计算每个高斯的贡献
    for i in range(len(centers)):
        center = centers[i]
        cov = covs[i]
        weight = weights[i]
        
        # 计算从当前高斯得到的分布值
        Z += weight * np.exp(-cov * ((X - center[0])**2 + (Y - center[1])**2))
    
    return Z