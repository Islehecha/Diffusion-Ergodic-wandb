import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.encoder import ErgodicEncoder
from .module.decoder import ErgodicDecoder
from .diffusion_utils.sde import VPSDE_linear


class ErgodicDiffusionModel(nn.Module):
    """
    简化的遍历扩散模型
    
    重构后的设计原则：
    1. 只使用x_start formulation，移除混合训练模式
    2. 纯扩散损失，移除训练时的约束项
    3. 简化配置和参数管理
    4. 约束处理移到独立的后处理阶段
    """
    
    def __init__(self, config):
        super().__init__()

        self.config = config

        # SDE配置
        self.sde = VPSDE_linear(
            beta_min=getattr(config.diffusion, 'beta_min', 0.1),
            beta_max=getattr(config.diffusion, 'beta_max', 20.0)
        )

        # 创建编码器和解码器
        self.encoder = ErgodicEncoder(config)
        self.decoder = ErgodicDecoder(config)

        # 固定使用x_start模式
        self.model_type = "x_start"
        self.decoder.model_type = self.model_type
        if hasattr(self.decoder, 'dit'):
            self.decoder.dit.model_type = self.model_type

        # 起点约束系数（简化版本）
        self.start_point_coef = getattr(config.diffusion, 'start_point_coef', 10.0)

        # 初始化损失组件记录
        self.last_loss_components = {}

    def _denorm_robot_state(self, rs: torch.Tensor) -> torch.Tensor:
        """将标准化的 robot_state 还原到物理坐标系（与轨迹同尺度）。
        需要 config.normalizer.robot_state.{mean,std}。
        """
        if rs is None:
            return None
        mean = self.config.normalizer.robot_state.mean.to(rs.device).view(1, -1)
        std = self.config.normalizer.robot_state.std.to(rs.device).view(1, -1)
        return rs * std + mean

    def forward(self, inputs, training=None):
        """
        简化的前向传播 - 只处理x_start模式
        
        Args:
            inputs: 字典，包含:
                - distribution: [B, 1, H, W] - 分布图
                - robot_state: [B, state_dim] - 机器人初始状态
                - trajectories: [B, trajectory_len, state_dim] - 目标轨迹 (训练时)
                - diffusion_time: [B] - 扩散时间步 (训练时)
        """
        # 编码输入
        encoder_inputs = {
            'distribution': inputs['distribution']
        }
        
        if 'robot_state' in inputs:
            encoder_inputs['robot_state'] = inputs['robot_state']
            
        encoder_outputs = self.encoder(encoder_inputs)
        
        # 解码器输入
        decoder_inputs = {
            'robot_state': inputs.get('robot_state'),
            'trajectories': inputs.get('trajectories'),
            'diffusion_time': inputs.get('diffusion_time')
        }

        # 构造硬性条件字典：强制第 0 个时间步等于给定的初始状态（使用反标准化物理坐标）
        conditions = {}
        if inputs.get('robot_state') is not None:
            conditions[0] = self._denorm_robot_state(inputs['robot_state'])  # [B, state_dim]
        # 如需终点条件，可在此添加：
        # if inputs.get('trajectories') is not None:
        #     conditions[self.config.trajectory_len - 1] = inputs['trajectories'][:, -1, :]
        if len(conditions) > 0:
            decoder_inputs['conditions'] = conditions

        # 解码器前向传播
        outputs = self.decoder(encoder_outputs, decoder_inputs)

        # 简化输出 - 只返回扩散相关的结果
        return {
            'prediction': outputs.get('score', outputs.get('prediction')),  # x_start预测
            'diffusion_time': inputs.get('diffusion_time'),
            'robot_state': inputs.get('robot_state')
        }

    def compute_loss(self, outputs, target_trajectories):
        """
        纯扩散损失函数 - 移除所有约束项
        
        Args:
            outputs: 模型输出
            target_trajectories: 目标轨迹 [B, trajectory_len, state_dim]
        
        Returns:
            torch.Tensor: 纯扩散损失
        """
        self.last_loss_components = {}
        
        # 获取预测结果
        prediction = outputs['prediction']
        robot_state = outputs.get('robot_state')
        
        # 1. 主要损失：MSE损失（x_start formulation）
        mse_loss = F.mse_loss(prediction, target_trajectories)
        self.last_loss_components['mse_loss'] = mse_loss.item()
        
        # 总损失（移除起点软惩罚项）
        total_loss = mse_loss
        self.last_loss_components['total_loss'] = total_loss.item()
        return total_loss

    def inference(self, inputs):
        """
        简化的推理函数
        
        Args:
            inputs: dict
                'distribution': [B, 1, H, W] - 二维空间分布
                'robot_state': [B, state_dim] - 机器人初始状态
        
        Returns:
            dict:
                'prediction': [B, T, state_dim] - 生成的轨迹
        """
        self.eval()
        
        with torch.no_grad():
            # 编码（编码器继续使用标准化的 robot_state 作为条件）
            encoder_outputs = self.encoder(inputs)

            # 解码生成轨迹：为解码器提供“反标准化”的起点条件
            inputs_for_decoder = dict(inputs)
            if inputs.get('robot_state') is not None:
                inputs_for_decoder['robot_state'] = self._denorm_robot_state(inputs['robot_state'])

            outputs = self.decoder.inference(encoder_outputs, inputs_for_decoder)

            return {
                'prediction': outputs.get('trajectories', outputs.get('prediction'))
            }
    
    def get_loss_components(self):
        """获取损失组件用于监控"""
        return self.last_loss_components.copy()
    
    def compute_constraint_metrics(self, trajectories):
        """
        计算约束指标用于监控（不参与训练）
        
        Args:
            trajectories: [B, T, state_dim] - 轨迹数据
            
        Returns:
            Dict[str, float]: 约束指标
        """
        metrics = {}
        
        # 提取位置信息
        positions = trajectories[:, :, :2]  # [B, T, 2]
        
        # 计算运动学量
        velocity = positions[:, 1:] - positions[:, :-1]  # [B, T-1, 2]
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 2]
        
        # 平滑度指标
        velocity_magnitude = torch.norm(velocity, dim=2)
        acceleration_magnitude = torch.norm(acceleration, dim=2)
        
        metrics['avg_velocity'] = velocity_magnitude.mean().item()
        metrics['avg_acceleration'] = acceleration_magnitude.mean().item()
        
        # 可行性指标（简化版）
        max_velocity = 1.0  # 可以从配置读取
        max_acceleration = 2.0
        delta_t = 0.1
        
        velocity_real = velocity_magnitude / delta_t
        acceleration_real = acceleration_magnitude / (delta_t ** 2)
        
        velocity_violations = torch.clamp(velocity_real - max_velocity, min=0.0)
        acceleration_violations = torch.clamp(acceleration_real - max_acceleration, min=0.0)
        
        metrics['velocity_violation_rate'] = (velocity_violations > 0).float().mean().item()
        metrics['acceleration_violation_rate'] = (acceleration_violations > 0).float().mean().item()
        
        return metrics