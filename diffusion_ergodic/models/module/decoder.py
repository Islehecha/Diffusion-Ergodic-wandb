import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from ..diffusion_utils.sampling import dpm_sampler
from .dit import TimestepEmbedder, DiTBlock, FinalLayer

class ErgodicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.trajectory_len = getattr(config, 'trajectory_len', getattr(config.data, 'trajectory_len', 40))
        self.robot_state_dim = getattr(config, 'robot_state_dim', getattr(config.data, 'robot_state_dim', 4))
        self.output_dim = self.trajectory_len * self.robot_state_dim  # 完整轨迹维度
        self.hidden_dim = getattr(config, 'hidden_dim', getattr(config.model, 'hidden_dim', 192))
        self.model_type = getattr(config, 'diffusion_model_type', getattr(config.diffusion, 'model_type', "x_start"))

        # 创建 DiT 模型
        self.dit = ErgodicDiT(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            depth=getattr(config, 'decoder_depth', getattr(config.model, 'decoder_depth', 3)),
            heads=getattr(config, 'num_heads', getattr(config.model, 'num_heads', 6)),
            dropout=getattr(config, 'decoder_drop_path_rate', getattr(config.model, 'decoder_drop_path_rate', 0.1)),
            model_type=self.model_type
        )

    def forward(self, encoder_outputs, inputs):
        """训练阶段，预测轨迹的分数或去噪轨迹"""
        encoding = encoder_outputs['encoding']  # [B, hidden_dim]

        # 从encoder_outputs中获取robot_state，如果存在的话
        robot_state = encoder_outputs.get('robot_state')

        # 如果有训练所需的输入，无论训练状态如何，都执行训练逻辑
        if 'trajectories' in inputs and 'diffusion_time' in inputs:
            # 训练模式
            trajectories = inputs['trajectories']  # [B, trajectory_len, state_dim]
            B = trajectories.shape[0]
            trajectories = trajectories.reshape(B, -1)  # 展平为 [B, trajectory_len*state_dim]

            diffusion_time = inputs['diffusion_time']  # [B]

            # 确保输出维度正确
            # 传递硬性条件（如起点）
            conditions = inputs.get('conditions') if isinstance(inputs, dict) else None
            output = self.dit(
                trajectories,
                diffusion_time,
                encoding,
                conditions=conditions
            )

            # 重塑输出为 [B, trajectory_len, state_dim]
            output = output.reshape(B, self.trajectory_len, self.robot_state_dim)

            # 添加模型类型的输出解释说明
            result = {
                "score": output,
                "diffusion_time": diffusion_time
            }

            # 从inputs中获取robot_state，如果存在的话
            if 'robot_state' in inputs:
                robot_state = inputs['robot_state']

            # 如果仍然没有robot_state，尝试从trajectories中获取
            if robot_state is None and 'trajectories' in inputs:
                print("[DEBUG] Extracting robot_state from trajectories first point in decoder")
                robot_state = inputs['trajectories'][:, 0, :]

            # 添加robot_state用于物理约束计算
            if robot_state is not None:
                result['robot_state'] = robot_state
                # print(f"[DEBUG] Added robot_state to decoder output: {robot_state.shape}")
            else:
                print("[WARNING] robot_state is still None in decoder output")

            return result
        else:
            # 如果缺少必要输入，返回空字典
            return {}

    def inference(self, encoder_outputs, inputs):
        """推理阶段，生成轨迹"""
        encoding = encoder_outputs['encoding']  # [B, hidden_dim]
        B = encoding.shape[0]

        # 初始化为高斯噪声
        x_T = torch.randn(B, self.output_dim, device=encoding.device)

        # 准备模型参数
        other_model_params = {
            'context': encoding  # 传递编码输出作为上下文
        }
        # 将硬性条件传递给采样器模型
        if isinstance(inputs, dict) and 'robot_state' in inputs and inputs['robot_state'] is not None:
            other_model_params['conditions'] = {0: inputs['robot_state']}

        # 获取扩散步数
        diffusion_steps = getattr(self.config, 'diffusion_steps',
                                getattr(self.config.diffusion, 'steps', 20))

        # 调用dpm_sampler，与它的函数签名保持一致
        x0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params=other_model_params,
            diffusion_steps=diffusion_steps
        )

        # 重塑为轨迹形状
        trajectories = x0.reshape(B, self.trajectory_len, self.robot_state_dim)

        # 强制起点与给定初始状态一致（物理坐标）。注意：inputs['robot_state'] 此时已是反标准化版本
        if isinstance(inputs, dict) and 'robot_state' in inputs and inputs['robot_state'] is not None:
            trajectories[:, 0, :] = inputs['robot_state']

        return {
            "trajectories": trajectories
        }


class ErgodicDiT(nn.Module):
    """
    Diffusion Transformer for Ergodic trajectories
    """
    def __init__(self, output_dim, hidden_dim, depth, heads, dropout=0.1, model_type="x_start"):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        # 轨迹投影
        self.traj_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # DiT模块
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(depth)
        ])

        # 输出层
        self.output_layer = FinalLayer(hidden_dim, output_dim)

    def forward(self, x, t, context, conditions=None):
        """
        Args:
            x: [B, output_dim] - 轨迹
            t: [B] - 扩散时间步
            context: [B, hidden_dim] - 编码器输出的条件

        Returns:
            # 根据模型类型，输出有不同的解释：
            # - 如果 model_type == "x_start"，输出是预测的去噪轨迹
            # - 如果 model_type == "score"，输出是预测的得分（负归一化噪声）
        """
        # 硬性起点约束：每次前向都将第 0 时刻强制设为给定初始状态
        if conditions is not None and isinstance(conditions, dict) and 0 in conditions and conditions[0] is not None:
            B = x.shape[0]
            D = conditions[0].shape[-1]
            T = self.output_dim // D
            x_view = x.view(B, T, D)
            x_view[:, 0, :] = conditions[0].to(x_view.device)
            x = x_view.view(B, self.output_dim)

        # 投影轨迹
        x_proj = self.traj_proj(x)  # [B, hidden_dim]

        # 时间嵌入 - 只传入 t
        t_emb = self.t_embedder(t)  # [B, hidden_dim]

        # 将条件、时间和投影的轨迹结合
        h = x_proj + t_emb  # [B, hidden_dim]

        # 应用DiT块
        for block in self.dit_blocks:
            h = block(h, context)

        # 输出层
        output = self.output_layer(h, t_emb)

        return output