"""
Diffusion-Ergodic Model Deployment Interface

Standalone deployment script for the Diffusion-Ergodic model.
Provides a clean interface for loading trained models and performing inference
on deployment platforms with minimal dependencies.

Usage:
    interface = DiffusionErgodicInterface(model_path, config_path)
    trajectories = interface.generate_trajectory(distribution, robot_state)
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union, Optional, Any
import warnings


class DiffusionErgodicInterface:
    """
    Deployment interface for the Diffusion-Ergodic model.
    
    This class provides a simplified interface for loading and using
    the trained diffusion model for ergodic trajectory generation.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the deployment interface.
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            config_path: Optional path to config file. If None, attempts to load
                        from the same directory as model_path
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.normalizer_params = None
        
        # Load configuration
        if config_path is None:
            config_dir = os.path.dirname(model_path)
            config_path = os.path.join(config_dir, 'model_config.yaml')
        
        self._load_config(config_path)
        self._load_model(model_path)
        self._setup_normalizer()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Convert dict to object-like access for compatibility
            self.config = self._dict_to_obj(self.config)
            
        except FileNotFoundError:
            warnings.warn(f"Config file not found: {config_path}. Using default config.")
            self.config = self._get_default_config()
    
    def _dict_to_obj(self, d):
        """Convert nested dict to object with attribute access."""
        if isinstance(d, dict):
            obj = type('Config', (), {})()
            for k, v in d.items():
                setattr(obj, k, self._dict_to_obj(v))
            return obj
        return d
    
    def _get_default_config(self):
        """Get default configuration if config file is not available."""
        config_dict = {
            'data': {
                'trajectory_len': 40,
                'robot_state_dim': 4,
                'distribution_dim': [32, 32]
            },
            'model': {
                'encoder_depth': 3,
                'decoder_depth': 6,
                'num_heads': 6,
                'hidden_dim': 384
            },
            'diffusion': {
                'model_type': 'x_start',
                'steps': 20,
                'beta_min': 0.1,
                'beta_max': 20.0
            },
            'normalizer': {
                'robot_state': {
                    'mean': [0.5, 0.5, 0.0, 0.0],
                    'std': [0.5, 0.5, 3.14, 1.0]
                }
            }
        }
        return self._dict_to_obj(config_dict)
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model instance
        self.model = ErgodicDiffusionModel(self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_normalizer(self):
        """Setup normalization parameters."""
        if hasattr(self.config, 'normalizer') and hasattr(self.config.normalizer, 'robot_state'):
            self.normalizer_params = {
                'mean': torch.tensor(self.config.normalizer.robot_state.mean, 
                                   dtype=torch.float32, device=self.device),
                'std': torch.tensor(self.config.normalizer.robot_state.std, 
                                  dtype=torch.float32, device=self.device)
            }
        else:
            # Default normalization parameters
            self.normalizer_params = {
                'mean': torch.tensor([0.5, 0.5, 0.0, 0.0], 
                                   dtype=torch.float32, device=self.device),
                'std': torch.tensor([0.5, 0.5, 3.14, 1.0], 
                                  dtype=torch.float32, device=self.device)
            }
    
    def _preprocess_distribution(self, distribution: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """
        Preprocess input distribution to the expected format.
        
        Args:
            distribution: Input distribution, can be:
                - numpy array of shape [H, W] or [1, H, W] or [B, 1, H, W]
                - torch tensor of same shapes
                - nested list that can be converted to array
        
        Returns:
            torch.Tensor: Preprocessed distribution of shape [B, 1, H, W]
        """
        # Convert to tensor if needed
        if isinstance(distribution, (list, np.ndarray)):
            distribution = torch.tensor(distribution, dtype=torch.float32)
        elif not isinstance(distribution, torch.Tensor):
            raise TypeError(f"Unsupported distribution type: {type(distribution)}")
        
        # Ensure correct shape
        if distribution.dim() == 2:  # [H, W]
            distribution = distribution.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif distribution.dim() == 3:  # [1, H, W] or [B, H, W]
            if distribution.shape[0] == 1:  # [1, H, W]
                distribution = distribution.unsqueeze(0)  # [1, 1, H, W]
            else:  # [B, H, W]
                distribution = distribution.unsqueeze(1)  # [B, 1, H, W]
        elif distribution.dim() != 4:
            raise ValueError(f"Invalid distribution shape: {distribution.shape}")
        
        # Normalize using max-value normalization (critical for consistency)
        distribution = distribution / (distribution.max() + 1e-8)
        
        # Move to device
        distribution = distribution.to(self.device)
        
        return distribution
    
    def _preprocess_robot_state(self, robot_state: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """
        Preprocess robot state to the expected format.
        
        Args:
            robot_state: Robot state [x, y, theta, v] or batch of states
        
        Returns:
            torch.Tensor: Preprocessed robot state of shape [B, state_dim]
        """
        # Convert to tensor if needed
        if isinstance(robot_state, (list, np.ndarray)):
            robot_state = torch.tensor(robot_state, dtype=torch.float32)
        elif not isinstance(robot_state, torch.Tensor):
            raise TypeError(f"Unsupported robot_state type: {type(robot_state)}")
        
        # Ensure batch dimension
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)  # [1, state_dim]
        
        # Move to device
        robot_state = robot_state.to(self.device)
        
        # Check if normalization is needed (heuristic: if values are not in normalized range)
        if self._needs_normalization(robot_state):
            robot_state = self._normalize_robot_state(robot_state)
        
        return robot_state
    
    def _needs_normalization(self, robot_state: torch.Tensor) -> bool:
        """
        Heuristic to determine if robot state needs normalization.
        Assumes normalized data should be roughly in [-2, 2] range.
        """
        return torch.any(torch.abs(robot_state) > 2.5)
    
    def _normalize_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Normalize robot state using stored parameters."""
        if self.normalizer_params is not None:
            return (robot_state - self.normalizer_params['mean']) / self.normalizer_params['std']
        return robot_state
    
    def _denormalize_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Denormalize robot state using stored parameters."""
        if self.normalizer_params is not None:
            return robot_state * self.normalizer_params['std'] + self.normalizer_params['mean']
        return robot_state
    
    def generate_trajectory(self, 
                          distribution: Union[np.ndarray, torch.Tensor, list],
                          robot_state: Union[np.ndarray, torch.Tensor, list],
                          denormalize_output: bool = True) -> np.ndarray:
        """
        Generate ergodic trajectory for given distribution and robot state.
        
        Args:
            distribution: Target distribution to cover, shape [H, W] or compatible
            robot_state: Initial robot state [x, y, theta, v] or compatible
            denormalize_output: Whether to denormalize the output trajectory
        
        Returns:
            numpy.ndarray: Generated trajectory of shape [trajectory_len, state_dim]
        """
        # Preprocess inputs
        dist_tensor = self._preprocess_distribution(distribution)
        robot_tensor = self._preprocess_robot_state(robot_state)
        
        # Prepare model inputs
        inputs = {
            'distribution': dist_tensor,
            'robot_state': robot_tensor
        }
        
        # Generate trajectory
        with torch.no_grad():
            outputs = self.model.inference(inputs)
            trajectory = outputs['prediction']  # [B, T, state_dim]
        
        # Remove batch dimension if single input
        if trajectory.shape[0] == 1:
            trajectory = trajectory.squeeze(0)  # [T, state_dim]
        
        # Denormalize if requested
        if denormalize_output:
            trajectory = self._denormalize_robot_state(trajectory)
        
        # Convert to numpy
        return trajectory.cpu().numpy()
    
    def generate_multiple_trajectories(self,
                                     distribution: Union[np.ndarray, torch.Tensor, list],
                                     robot_states: Union[np.ndarray, torch.Tensor, list],
                                     denormalize_output: bool = True) -> np.ndarray:
        """
        Generate multiple trajectories for batch of robot states.
        
        Args:
            distribution: Target distribution to cover
            robot_states: Batch of initial robot states [B, state_dim]
            denormalize_output: Whether to denormalize the output trajectories
        
        Returns:
            numpy.ndarray: Generated trajectories of shape [B, trajectory_len, state_dim]
        """
        # Preprocess inputs
        dist_tensor = self._preprocess_distribution(distribution)
        robot_tensor = self._preprocess_robot_state(robot_states)
        
        # Ensure distribution matches batch size
        if dist_tensor.shape[0] == 1 and robot_tensor.shape[0] > 1:
            dist_tensor = dist_tensor.expand(robot_tensor.shape[0], -1, -1, -1)
        
        # Prepare model inputs
        inputs = {
            'distribution': dist_tensor,
            'robot_state': robot_tensor
        }
        
        # Generate trajectories
        with torch.no_grad():
            outputs = self.model.inference(inputs)
            trajectories = outputs['prediction']  # [B, T, state_dim]
        
        # Denormalize if requested
        if denormalize_output:
            trajectories = self._denormalize_robot_state(trajectories)
        
        # Convert to numpy
        return trajectories.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'trajectory_length': self.config.data.trajectory_len,
            'robot_state_dim': self.config.data.robot_state_dim,
            'distribution_dim': self.config.data.distribution_dim,
            'model_type': self.config.diffusion.model_type,
            'diffusion_steps': self.config.diffusion.steps,
            'device': str(self.device),
            'normalizer_params': {
                'mean': self.normalizer_params['mean'].cpu().tolist(),
                'std': self.normalizer_params['std'].cpu().tolist()
            } if self.normalizer_params else None
        }


# Minimal model implementations for deployment
class ErgodicDiffusionModel(nn.Module):
    """Simplified ErgodicDiffusionModel for deployment."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ErgodicEncoder(config)
        self.decoder = ErgodicDecoder(config)
        
        # Import SDE here to avoid circular imports
        from sde import VPSDE_linear
        self.sde = VPSDE_linear(
            beta_min=getattr(config.diffusion, 'beta_min', 0.1),
            beta_max=getattr(config.diffusion, 'beta_max', 20.0)
        )
    
    def inference(self, inputs):
        """Inference method for deployment."""
        self.eval()
        with torch.no_grad():
            encoder_outputs = self.encoder(inputs)
            outputs = self.decoder.inference(encoder_outputs, inputs)
            return {'prediction': outputs.get('trajectories', outputs.get('prediction'))}


class ErgodicEncoder(nn.Module):
    """Simplified ErgodicEncoder for deployment."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.model.hidden_dim
        robot_state_dim = config.data.robot_state_dim
        distribution_dim = config.data.distribution_dim
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Distribution encoder
        self.distribution_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.GELU()
        )
        
        # Distribution projector
        dist_h = distribution_dim[0] // 4
        dist_w = distribution_dim[1] // 4
        dist_feature_size = dist_h * dist_w * 64
        self.distribution_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dist_feature_size, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, inputs_dict):
        robot_state = inputs_dict['robot_state']
        distribution = inputs_dict['distribution']
        
        robot_encoding = self.robot_encoder(robot_state)
        dist_features = self.distribution_encoder(distribution)
        distribution_encoding = self.distribution_projector(dist_features)
        
        encoding = robot_encoding + distribution_encoding
        encoding = self.fusion(encoding)
        
        return {
            'encoding': encoding,
            'robot_state': robot_state
        }


class ErgodicDecoder(nn.Module):
    """Simplified ErgodicDecoder for deployment."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trajectory_len = config.data.trajectory_len
        self.robot_state_dim = config.data.robot_state_dim
        self.output_dim = self.trajectory_len * self.robot_state_dim
        self.hidden_dim = config.model.hidden_dim
        
        self.dit = ErgodicDiT(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            depth=config.model.decoder_depth,
            heads=config.model.num_heads,
            dropout=0.1,
            model_type=config.diffusion.model_type
        )
    
    def inference(self, encoder_outputs, inputs):
        """Inference for trajectory generation."""
        encoding = encoder_outputs['encoding']
        B = encoding.shape[0]
        
        # Initialize with Gaussian noise
        x_T = torch.randn(B, self.output_dim, device=encoding.device)
        
        # Prepare model parameters
        other_model_params = {'context': encoding}
        
        # Get diffusion steps
        diffusion_steps = getattr(self.config.diffusion, 'steps', 20)
        
        # Use DPM sampler
        from dpm_sampler import dpm_sampler
        x0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params=other_model_params,
            diffusion_steps=diffusion_steps
        )
        
        # Reshape to trajectory format
        trajectories = x0.reshape(B, self.trajectory_len, self.robot_state_dim)
        
        return {'trajectories': trajectories}


class ErgodicDiT(nn.Module):
    """Simplified Diffusion Transformer for deployment."""
    
    def __init__(self, output_dim, hidden_dim, depth, heads, dropout=0.1, model_type="x_start"):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        # Trajectory projection
        self.traj_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Time embedder
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Output layer
        self.output_layer = FinalLayer(hidden_dim, output_dim)
    
    def forward(self, x, t, context):
        x_proj = self.traj_proj(x)
        t_emb = self.t_embedder(t)
        
        h = x_proj + t_emb
        
        for block in self.dit_blocks:
            h = block(h, context)
        
        output = self.output_layer(h, t_emb)
        return output


# Import necessary components from the main codebase
# These are minimal implementations to avoid circular imports

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention block
        attn_input = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP block  
        mlp_input = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


# The actual DPM sampler is imported from dmp_sampler.py module


def _simple_euler_sampling(model, x_T, other_model_params, steps):
    """Simple Euler sampling as fallback."""
    x = x_T
    dt = 1.0 / steps
    
    for i in range(steps):
        t = torch.ones(x.shape[0], device=x.device) * (1.0 - i * dt)
        with torch.no_grad():
            if other_model_params:
                pred = model(x, t, other_model_params['context'])
            else:
                pred = model(x, t, None)
            x = x - dt * pred
    
    return x


if __name__ == "__main__":
    # Example usage
    model_path = "./trained/best_model.pth"
    config_path = "./trained/model_config.yaml"
    
    try:
        # Initialize interface
        interface = DiffusionErgodicInterface(model_path, config_path)
        
        # Print model info
        print("Model Info:")
        info = interface.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Example inference
        distribution = np.random.rand(32, 32)  # Example distribution
        robot_state = [1.0, 1.0, 0.0, 0.5]    # Example initial state
        
        trajectory = interface.generate_trajectory(distribution, robot_state)
        print(f"\nGenerated trajectory shape: {trajectory.shape}")
        print(f"First few points: {trajectory[:3]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure model and config files exist in the expected locations.")