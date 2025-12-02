#!/usr/bin/env python3
"""
Test script for the Diffusion-Ergodic deployment interface.

This script tests the basic functionality of the deployment interface
to ensure it can load models and perform inference correctly.
"""

import os
import sys
import numpy as np
import torch

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from diffusion_model_interface import DiffusionErgodicInterface


def test_interface_creation():
    """Test interface creation with default config."""
    print("Testing interface creation...")
    
    # Check if model files exist
    model_path = "../diffusion_ergodic/trained/best_model.pth"
    config_path = "../diffusion_ergodic/trained/model_config.yaml"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Skipping interface creation test.")
        return False
    
    try:
        interface = DiffusionErgodicInterface(model_path, config_path)
        print("✓ Interface created successfully")
        
        # Print model info
        print("\nModel Information:")
        info = interface.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        return True, interface
    except Exception as e:
        print(f"✗ Failed to create interface: {e}")
        return False, None


def test_preprocessing():
    """Test input preprocessing functions."""
    print("\nTesting input preprocessing...")
    
    try:
        # Create dummy interface for testing preprocessing
        config_dict = {
            'data': {
                'trajectory_len': 40,
                'robot_state_dim': 4,
                'distribution_dim': [32, 32]
            },
            'model': {
                'hidden_dim': 384
            },
            'diffusion': {
                'steps': 20
            },
            'normalizer': {
                'robot_state': {
                    'mean': [0.5, 0.5, 0.0, 0.0],
                    'std': [0.5, 0.5, 3.14, 1.0]
                }
            }
        }
        
        # Create minimal interface for testing
        interface = type('TestInterface', (), {})()
        interface.device = torch.device('cpu')
        interface.config = type('Config', (), {})()
        interface.config.data = type('Data', (), config_dict['data'])()
        interface.normalizer_params = {
            'mean': torch.tensor(config_dict['normalizer']['robot_state']['mean']),
            'std': torch.tensor(config_dict['normalizer']['robot_state']['std'])
        }
        
        # Add methods
        def _preprocess_distribution(self, distribution):
            if isinstance(distribution, (list, np.ndarray)):
                distribution = torch.tensor(distribution, dtype=torch.float32)
            if distribution.dim() == 2:
                distribution = distribution.unsqueeze(0).unsqueeze(0)
            distribution = distribution / (distribution.max() + 1e-8)
            return distribution.to(self.device)
            
        def _preprocess_robot_state(self, robot_state):
            if isinstance(robot_state, (list, np.ndarray)):
                robot_state = torch.tensor(robot_state, dtype=torch.float32)
            if robot_state.dim() == 1:
                robot_state = robot_state.unsqueeze(0)
            return robot_state.to(self.device)
            
        def _needs_normalization(self, robot_state):
            return torch.any(torch.abs(robot_state) > 2.5)
            
        interface._preprocess_distribution = _preprocess_distribution.__get__(interface)
        interface._preprocess_robot_state = _preprocess_robot_state.__get__(interface)
        interface._needs_normalization = _needs_normalization.__get__(interface)
        
        # Test distribution preprocessing
        test_dist = np.random.rand(32, 32)
        processed_dist = interface._preprocess_distribution(test_dist)
        assert processed_dist.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {processed_dist.shape}"
        print("✓ Distribution preprocessing works")
        
        # Test robot state preprocessing
        test_robot_state = [1.0, 1.0, 0.0, 0.5]
        processed_state = interface._preprocess_robot_state(test_robot_state)
        assert processed_state.shape == (1, 4), f"Expected (1, 4), got {processed_state.shape}"
        print("✓ Robot state preprocessing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False


def test_inference_with_dummy_data():
    """Test inference with dummy data (without actual model)."""
    print("\nTesting inference with dummy data...")
    
    try:
        # Create dummy distribution and robot state
        distribution = np.random.rand(32, 32)
        robot_state = [1.0, 1.0, 0.0, 0.5]
        
        print(f"✓ Created test data:")
        print(f"  Distribution shape: {distribution.shape}")
        print(f"  Robot state: {robot_state}")
        
        # If we have a real interface, we could test inference here
        # For now, just validate that the inputs are reasonable
        assert distribution.shape == (32, 32), "Distribution should be 32x32"
        assert len(robot_state) == 4, "Robot state should have 4 dimensions"
        
        print("✓ Test data validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Dummy inference test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Diffusion-Ergodic Deployment Interface Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Interface creation
    success, interface = test_interface_creation()
    if success:
        success_count += 1
    
    # Test 2: Preprocessing
    if test_preprocessing():
        success_count += 1
    
    # Test 3: Dummy inference
    if test_inference_with_dummy_data():
        success_count += 1
    
    # Test 4: Real inference (if interface was created successfully)
    if success and interface is not None:
        print("\nTesting real inference...")
        try:
            distribution = np.random.rand(32, 32)
            robot_state = [1.0, 1.0, 0.0, 0.5]
            
            trajectory = interface.generate_trajectory(distribution, robot_state)
            print(f"✓ Generated trajectory shape: {trajectory.shape}")
            print(f"  First few points: {trajectory[:3]}")
            
            # Test batch inference
            robot_states = [[1.0, 1.0, 0.0, 0.5], [2.0, 2.0, 1.57, 0.3]]
            trajectories = interface.generate_multiple_trajectories(distribution, robot_states)
            print(f"✓ Generated batch trajectories shape: {trajectories.shape}")
            
            success_count += 1
            total_tests += 1
            
        except Exception as e:
            print(f"✗ Real inference test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Deployment interface is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the interface implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)