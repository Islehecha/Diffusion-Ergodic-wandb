import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsConstraintProcessor:
    """
    物理约束处理器 - 独立于扩散模型的约束处理
    
    设计原则：
    1. 不参与扩散模型训练
    2. 在推理时后处理应用约束
    3. 可以独立调整约束强度
    4. 支持多种约束类型
    """
    
    def __init__(self, config):
        """
        初始化约束处理器
        
        Args:
            config: 配置对象，包含约束相关参数
        """
        self.config = config
        
        # 从配置或使用默认值
        constraints_config = getattr(config, 'constraints', {})
        
        # 物理参数
        self.max_velocity = getattr(constraints_config, 'max_velocity', 1.0)
        self.max_acceleration = getattr(constraints_config, 'max_acceleration', 2.0)
        self.delta_t = getattr(constraints_config, 'delta_t', 0.1)
        
        # 边界约束
        self.min_position = torch.tensor(getattr(constraints_config, 'min_position', [0.0, -1.0]))
        self.max_position = torch.tensor(getattr(constraints_config, 'max_position', [3.5, 3.5]))
        
        # 约束权重
        self.smoothness_weight = getattr(constraints_config, 'smoothness_weight', 0.1)
        self.feasibility_weight = getattr(constraints_config, 'feasibility_weight', 0.1)
        
        # 约束应用开关
        self.apply_smoothness = getattr(constraints_config, 'apply_smoothness', True)
        self.apply_feasibility = getattr(constraints_config, 'apply_feasibility', True)
        self.apply_boundary = getattr(constraints_config, 'apply_boundary', True)
        
        # 默认约束强度
        self.default_constraint_strength = getattr(constraints_config, 'default_constraint_strength', 0.5)
        
    def evaluate_constraints(self, trajectories: torch.Tensor) -> Dict[str, float]:
        """
        评估轨迹的约束违规情况（用于监控，不参与训练）
        
        Args:
            trajectories: [B, T, state_dim] - 轨迹数据
            
        Returns:
            Dict[str, float]: 各种约束指标
        """
        metrics = {}
        
        # 提取位置信息
        positions = trajectories[:, :, :2]  # [B, T, 2]
        
        # 计算运动学量
        velocity = positions[:, 1:] - positions[:, :-1]  # [B, T-1, 2]
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 2]
        jerk = acceleration[:, 1:] - acceleration[:, :-1]  # [B, T-3, 2]
        
        # 1. 平滑度指标
        if self.apply_smoothness:
            velocity_magnitude = torch.norm(velocity, dim=2)
            acceleration_magnitude = torch.norm(acceleration, dim=2)
            jerk_magnitude = torch.norm(jerk, dim=2)
            
            metrics['avg_velocity'] = velocity_magnitude.mean().item()
            metrics['avg_acceleration'] = acceleration_magnitude.mean().item()
            metrics['avg_jerk'] = jerk_magnitude.mean().item()
            
            # 计算方向变化（曲率指标）
            if velocity_magnitude.max() > 1e-6:
                velocity_norm = velocity / (torch.norm(velocity, dim=2, keepdim=True) + 1e-6)
                direction_change = torch.norm(velocity_norm[:, 1:] - velocity_norm[:, :-1], dim=2)
                metrics['avg_curvature'] = direction_change.mean().item()
        
        # 2. 可行性指标
        if self.apply_feasibility:
            # 速度违规
            velocity_real = torch.norm(velocity, dim=2) / self.delta_t
            velocity_violations = torch.clamp(velocity_real - self.max_velocity, min=0.0)
            metrics['velocity_violation_rate'] = (velocity_violations > 0).float().mean().item()
            metrics['max_velocity_violation'] = velocity_violations.max().item()
            
            # 加速度违规
            acceleration_real = torch.norm(acceleration, dim=2) / (self.delta_t ** 2)
            acceleration_violations = torch.clamp(acceleration_real - self.max_acceleration, min=0.0)
            metrics['acceleration_violation_rate'] = (acceleration_violations > 0).float().mean().item()
            metrics['max_acceleration_violation'] = acceleration_violations.max().item()
        
        # 3. 边界约束
        if self.apply_boundary:
            # 检查位置边界
            lower_violations = torch.clamp(self.min_position.to(positions.device) - positions, min=0.0)
            upper_violations = torch.clamp(positions - self.max_position.to(positions.device), min=0.0)
            
            total_boundary_violations = lower_violations.sum(dim=2) + upper_violations.sum(dim=2)
            metrics['boundary_violation_rate'] = (total_boundary_violations > 0).float().mean().item()
            metrics['max_boundary_violation'] = total_boundary_violations.max().item()
        
        return metrics
    
    def apply_constraints(self, trajectories: torch.Tensor, 
                         constraint_strength: float = None) -> torch.Tensor:
        """
        应用约束到轨迹（后处理方式）
        
        Args:
            trajectories: [B, T, state_dim] - 原始轨迹
            constraint_strength: 约束强度 (0.0 = 无约束, 1.0 = 完全约束)
            
        Returns:
            torch.Tensor: 约束后的轨迹
        """
        if constraint_strength is None:
            constraint_strength = self.default_constraint_strength
            
        if constraint_strength <= 0:
            return trajectories
            
        constrained_trajectories = trajectories.clone()
        
        # 1. 边界约束（硬约束）
        if self.apply_boundary:
            constrained_trajectories = self._apply_boundary_constraints(constrained_trajectories)
        
        # 2. 平滑度约束（软约束）
        if self.apply_smoothness:
            constrained_trajectories = self._apply_smoothness_constraints(
                constrained_trajectories, constraint_strength
            )
        
        # 3. 可行性约束（软约束）
        if self.apply_feasibility:
            constrained_trajectories = self._apply_feasibility_constraints(
                constrained_trajectories, constraint_strength
            )
        
        return constrained_trajectories
    
    def _apply_boundary_constraints(self, trajectories: torch.Tensor) -> torch.Tensor:
        """应用边界约束（硬约束）"""
        positions = trajectories[:, :, :2]
        
        # 限制位置在边界内
        positions = torch.clamp(
            positions,
            min=self.min_position.to(positions.device),
            max=self.max_position.to(positions.device)
        )
        
        trajectories[:, :, :2] = positions
        return trajectories
    
    def _apply_smoothness_constraints(self, trajectories: torch.Tensor, 
                                    strength: float) -> torch.Tensor:
        """应用平滑度约束（软约束）"""
        if strength <= 0:
            return trajectories
            
        positions = trajectories[:, :, :2]
        
        # 应用移动平均平滑
        window_size = max(1, int(3 * strength))
        if window_size > 1:
            # 创建平滑核
            kernel = torch.ones(window_size, device=positions.device) / window_size
            
            # 对每个维度应用1D卷积平滑
            for b in range(positions.shape[0]):
                for d in range(positions.shape[2]):
                    # 填充以保持序列长度
                    padded = F.pad(positions[b, :, d], (window_size//2, window_size//2), mode='replicate')
                    smoothed = F.conv1d(padded.unsqueeze(0).unsqueeze(0), 
                                      kernel.unsqueeze(0).unsqueeze(0), 
                                      padding=0)
                    positions[b, :, d] = smoothed.squeeze()
            
            trajectories[:, :, :2] = positions
        
        return trajectories
    
    def _apply_feasibility_constraints(self, trajectories: torch.Tensor, 
                                     strength: float) -> torch.Tensor:
        """应用可行性约束（软约束）"""
        if strength <= 0:
            return trajectories
            
        positions = trajectories[:, :, :2]
        
        # 计算速度
        velocity = positions[:, 1:] - positions[:, :-1]
        velocity_magnitude = torch.norm(velocity, dim=2)
        
        # 速度限制
        max_vel_real = self.max_velocity * self.delta_t
        velocity_scale = torch.clamp(max_vel_real / (velocity_magnitude + 1e-6), max=1.0)
        
        # 应用速度约束
        velocity_constrained = velocity * velocity_scale.unsqueeze(-1)
        
        # 重新构建轨迹
        new_positions = torch.zeros_like(positions)
        new_positions[:, 0] = positions[:, 0]  # 保持起点
        
        for t in range(1, positions.shape[1]):
            new_positions[:, t] = new_positions[:, t-1] + velocity_constrained[:, t-1]
        
        # 根据约束强度混合原始轨迹和约束轨迹
        trajectories[:, :, :2] = (1 - strength) * positions + strength * new_positions
        
        return trajectories
    
    def get_constraint_summary(self, trajectories: torch.Tensor) -> str:
        """获取约束评估摘要"""
        metrics = self.evaluate_constraints(trajectories)
        
        summary = "Constraint Evaluation Summary:\n"
        summary += f"  Smoothness:\n"
        summary += f"    - Avg Velocity: {metrics.get('avg_velocity', 0):.4f}\n"
        summary += f"    - Avg Acceleration: {metrics.get('avg_acceleration', 0):.4f}\n"
        summary += f"    - Avg Curvature: {metrics.get('avg_curvature', 0):.4f}\n"
        summary += f"  Feasibility:\n"
        summary += f"    - Velocity Violation Rate: {metrics.get('velocity_violation_rate', 0):.2%}\n"
        summary += f"    - Acceleration Violation Rate: {metrics.get('acceleration_violation_rate', 0):.2%}\n"
        summary += f"  Boundary:\n"
        summary += f"    - Boundary Violation Rate: {metrics.get('boundary_violation_rate', 0):.2%}\n"
        
        return summary
    
    def create_constraint_config(self, max_velocity=1.0, max_acceleration=2.0, 
                                delta_t=0.1, constraint_strength=0.5) -> Dict:
        """创建约束配置的辅助函数"""
        return {
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration,
            'delta_t': delta_t,
            'min_position': [0.0, -1.0],
            'max_position': [3.5, 3.5],
            'smoothness_weight': 0.1,
            'feasibility_weight': 0.1,
            'apply_smoothness': True,
            'apply_feasibility': True,
            'apply_boundary': True,
            'default_constraint_strength': constraint_strength
        }