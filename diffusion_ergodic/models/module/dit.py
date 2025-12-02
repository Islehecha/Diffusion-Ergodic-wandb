import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import Mlp

class TimestepEmbedder(nn.Module):
    """
    将扩散时间步映射为embedding
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.half_dim = hidden_dim // 2
        
        # 将嵌入维度正确设置为 hidden_dim
        self.emb = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, t):
        """
        Args:
            t: [B] - 批量的时间步
        
        Returns:
            embeddings: [B, hidden_dim] - 嵌入的时间步
        """
        # 计算频率因子
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=self.half_dim, device=t.device) / self.half_dim
        )
        
        # 将每个时间步拓展为频率向量
        args = t[:, None] * freqs[None, :]  # [B, half_dim]
        
        # 计算正弦和余弦
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, hidden_dim]
        
        # 通过线性层
        embedding = self.emb(embedding)
        
        return embedding
    
class DiTBlock(nn.Module):
    """
    Diffusion Transformer块（简化版，适用于小型模型）
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 交叉注意力
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        self.norm3 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, 
                       act_layer=nn.GELU, drop=dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None):
        # 自注意力
        residual = x
        x = self.norm1(x)
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        x = self.self_attn(x, x, x)[0].squeeze(1)
        x = residual + self.dropout(x)
        
        # 交叉注意力
        if context is not None:
            residual = x
            x = self.norm2(x)
            context = context.unsqueeze(1)  # [B, 1, hidden_dim]
            x = x.unsqueeze(1)  # [B, 1, hidden_dim]
            x = self.cross_attn(x, context, context)[0].squeeze(1)
            x = residual + self.dropout(x)
        
        # MLP
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        
        return x


class FinalLayer(nn.Module):
    """
    DiT最后一层输出预测
    """
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, time_emb):
        """
        Args:
            x: [B, hidden_dim]
            time_emb: [B, hidden_dim]
        
        Returns:
            output: [B, output_dim]
        """
        # 可以选择使用时间嵌入进行调制
        x = self.norm(x)
        x = self.proj(x)
        return x