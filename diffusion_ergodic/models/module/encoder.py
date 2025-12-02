import torch
import torch.nn as nn
import torch.nn.functional as F

class ErgodicEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_dim = config.hidden_dim
        
        # 编码机器人状态
        self.robot_encoder = nn.Sequential(
            nn.Linear(config.robot_state_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # 编码 Ergodic 分布
        self.distribution_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # 降采样
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 再次降采样
            nn.GELU()
        )
        
        # 分布特征处理
        dist_h = config.distribution_dim[0] // 4
        dist_w = config.distribution_dim[1] // 4
        dist_feature_size = dist_h * dist_w * 64
        self.distribution_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dist_feature_size, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 融合编码
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

    def forward(self, inputs):
        # 编码机器人状态
        robot_state = inputs['robot_state']  # [B, state_dim]
        robot_encoding = self.robot_encoder(robot_state)
        
        # 编码 Ergodic 分布
        distribution = inputs['distribution']  # [B, 1, H, W]
        dist_features = self.distribution_encoder(distribution)
        distribution_encoding = self.distribution_projector(dist_features)
        
        # 融合表示作为条件
        encoding = robot_encoding + distribution_encoding
        encoding = self.fusion(encoding)
        
        # 修改：将robot_state也添加到输出字典中
        return {
            "encoding": encoding,
            "robot_state": robot_state  # 添加这一行以传递robot_state
        }