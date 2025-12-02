# Diffusion-Ergodic Model Deployment Interface

This folder contains a standalone deployment interface for the Diffusion-Ergodic model, designed for production environments with minimal dependencies.

## Files

- `diffusion_model_interface.py` - Main deployment interface class
- `sde.py` - SDE (Stochastic Differential Equation) classes for diffusion
- `dpm_sampler.py` - Simplified DPM sampler for inference  
- `test_interface.py` - Test script to validate the interface
- `README.md` - This documentation file

## Dependencies

The deployment interface requires minimal dependencies:
- `torch` (PyTorch)
- `numpy`
- `yaml` (for configuration loading)

## Usage

### Basic Usage

```python
from diffusion_model_interface import DiffusionErgodicInterface
import numpy as np

# Initialize the interface
model_path = "path/to/trained/model.pth"
config_path = "path/to/model/config.yaml"  # optional

interface = DiffusionErgodicInterface(model_path, config_path)

# Generate a single trajectory
distribution = np.random.rand(32, 32)  # Target distribution to cover
robot_state = [1.0, 1.0, 0.0, 0.5]     # Initial state [x, y, theta, v]

trajectory = interface.generate_trajectory(distribution, robot_state)
print(f"Generated trajectory shape: {trajectory.shape}")  # (40, 4)
```

### Batch Inference

```python
# Generate multiple trajectories
robot_states = [
    [1.0, 1.0, 0.0, 0.5],
    [2.0, 2.0, 1.57, 0.3],
    [0.5, 0.5, -1.57, 0.8]
]

trajectories = interface.generate_multiple_trajectories(distribution, robot_states)
print(f"Batch trajectories shape: {trajectories.shape}")  # (3, 40, 4)
```

### Model Information

```python
# Get information about the loaded model
info = interface.get_model_info()
print("Model Info:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

## Input Formats

### Distribution
The target distribution can be provided in various formats:
- `numpy.ndarray` of shape `[H, W]`, `[1, H, W]`, or `[B, 1, H, W]`
- `torch.Tensor` of compatible shapes
- Nested Python list convertible to array

**Important**: The interface uses max-value normalization (`grid / grid.max()`) for consistency with training data.

### Robot State
Initial robot state `[x, y, theta, v]` can be:
- Python list: `[1.0, 1.0, 0.0, 0.5]`
- `numpy.ndarray` of shape `[4]` or `[B, 4]`
- `torch.Tensor` of compatible shapes

The interface automatically detects if normalization is needed based on value ranges.

## Output Format

Generated trajectories are returned as `numpy.ndarray`:
- Single trajectory: shape `[trajectory_len, state_dim]` (default: `[40, 4]`)
- Batch trajectories: shape `[batch_size, trajectory_len, state_dim]`

By default, outputs are denormalized to real-world coordinates. Set `denormalize_output=False` to get normalized outputs.

## Configuration

If no config file is provided, the interface uses default parameters:
- Trajectory length: 40 steps
- Robot state dimensions: 4 `[x, y, theta, v]`
- Distribution grid: 32×32
- Diffusion steps: 20
- Model type: x_start

## Testing

Run the test script to validate the interface:

```bash
cd deployment
python test_interface.py
```

The test script will:
1. Attempt to load a trained model (if available)
2. Test input preprocessing functions
3. Validate inference with dummy data
4. Perform real inference (if model is loaded)

## Model Files

The interface expects two files:
1. **Model checkpoint** (`.pth`): Contains the trained model weights
2. **Config file** (`.yaml`): Contains model architecture and training parameters

Default locations:
- Model: `../diffusion_ergodic/trained/best_model.pth`
- Config: `../diffusion_ergodic/trained/model_config.yaml`

## Architecture

The deployment interface implements simplified versions of:
- `ErgodicDiffusionModel`: Main diffusion model
- `ErgodicEncoder`: Encodes distributions and robot states
- `ErgodicDecoder`: Generates trajectories using DiT architecture
- `ErgodicDiT`: Diffusion Transformer for trajectory generation

## Error Handling

The interface includes robust error handling for:
- Missing model/config files
- Invalid input formats
- Device compatibility (CPU/GPU)
- Normalization detection

## Performance Notes

- GPU acceleration used automatically if available
- Batch processing supported for multiple trajectories
- Minimal memory footprint suitable for deployment
- No training dependencies required

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all files are in the same directory
2. **Model loading fails**: Check file paths and permissions
3. **Shape errors**: Verify input distribution is 32×32
4. **Device errors**: Ensure PyTorch CUDA setup if using GPU

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DIFFUSION_DEBUG=1
python your_script.py
```

## Example Scripts

See `test_interface.py` for complete usage examples and validation tests.

## Integration

The interface is designed to be easily integrated into:
- REST API services
- Robot control systems
- Real-time planning pipelines
- Batch processing workflows

For production deployment, consider:
- Error logging and monitoring
- Input validation and sanitization
- Model version management
- Performance metrics collection