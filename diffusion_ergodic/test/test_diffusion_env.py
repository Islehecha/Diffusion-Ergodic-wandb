import os
import sys
import numpy as np
import torch
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_ergodic.data_process.ergodic_processor import ErgodicDataset
from diffusion_ergodic.models.diffusion_ergodic import ErgodicDiffusionModel
from diffusion_ergodic.training import load_config


def test_dataset_loading():
    """测试数据集加载"""
    print("初始化数据集...")
    dataset = ErgodicDataset(data_dir='diffusion_ergodic/data/ergodic_dataset')
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("数据集为空，请检查数据文件")
        return
    
    # 检查索引文件
    index_path = os.path.join('diffusion_ergodic/data/ergodic_dataset', 'dataset_index.json')
    if os.path.exists(index_path):
        print("\n索引文件内容概述:")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            if isinstance(index_data, dict):
                for key, value in index_data.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} 项")
                    else:
                        print(f"  {key}: {value}")
            elif isinstance(index_data, list):
                print(f"  数据对数量: {len(index_data)}")
                if len(index_data) > 0:
                    print(f"  第一个数据对: {index_data[0]}")
    
    # 打印目录内容以进行调试
    print("\n分布目录内容:")
    dist_dir = os.path.join('diffusion_ergodic/data/ergodic_dataset', 'distributions')
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        print(f"  文件数: {len(files)}")
        for f in files[:5]:  # 只显示前5个文件
            print(f"  - {f}")
            
            # 尝试读取第一个文件的内容
            if f.endswith('.json'):
                try:
                    with open(os.path.join(dist_dir, f), 'r') as file:
                        data = json.load(file)
                        print(f"    文件格式: {list(data.keys())}")
                except Exception as e:
                    print(f"    读取失败: {e}")
                break
    else:
        print(f"  目录 {dist_dir} 不存在")
    
    print("\n轨迹目录内容:")
    traj_dir = os.path.join('diffusion_ergodic/data/ergodic_dataset', 'trajectories')
    if os.path.exists(traj_dir):
        files = os.listdir(traj_dir)
        print(f"  文件数: {len(files)}")
        for f in files[:5]:  # 只显示前5个文件
            print(f"  - {f}")
            
            # 尝试读取第一个文件的内容
            if f.endswith('.json'):
                try:
                    with open(os.path.join(traj_dir, f), 'r') as file:
                        data = json.load(file)
                        print(f"    文件格式: {list(data.keys())}")
                except Exception as e:
                    print(f"    读取失败: {e}")
                break
    else:
        print(f"  目录 {traj_dir} 不存在")
    
    # 尝试获取第一个样本
    if len(dataset) > 0:
        try:
            print("\n获取第一个样本...")
            sample = dataset[0]
            
            # 打印样本信息
            print(f"样本内容:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 形状 {value.shape}, 类型 {value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            # 可视化样本
            print("\n可视化样本...")
            dataset.visualize_sample(0)
        except Exception as e:
            print(f"获取样本出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("数据集为空，无法获取样本")
    

def test_model_architecture():
    """测试模型架构"""
    # 加载配置
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        # 这里可以添加一些创建默认配置文件的逻辑
        config = create_default_config()
    else:
        config = load_config(config_path)
    
    print("配置内容:")
    print(f"  beta_min: {config.beta_min}")
    print(f"  beta_max: {config.beta_max}")
    print(f"  diffusion_model_type: {config.diffusion_model_type}")
    print(f"  hidden_dim: {config.hidden_dim}")
    
    # 创建模型
    try:
        model = ErgodicDiffusionModel(config)
        print("成功创建模型!")
        
        # 打印模型结构
        print("\n模型结构:")
        print(model)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params}")
        print(f"可训练参数数量: {trainable_params}")
    except Exception as e:
        print(f"创建模型失败: {e}")
        import traceback
        traceback.print_exc()

def create_default_config():
    """创建默认配置"""
    # 如果配置文件不存在，创建一个默认配置
    config_dict = {
        "data": {
            "data_dir": "diffusion_ergodic/data/ergodic_dataset",
            "trajectory_len": 100,
            "robot_state_dim": 4,
            "distribution_dim": [32, 32],
            "validation_split": 0.1,
            "shuffle_dataset": True,
            "num_workers": 4,
            "seed": 42
        },
        "model": {
            "encoder_depth": 3,
            "decoder_depth": 3,
            "num_heads": 6,
            "hidden_dim": 192,
            "encoder_drop_path_rate": 0.1,
            "decoder_drop_path_rate": 0.1
        },
        "diffusion": {
            "model_type": "x_start",
            "steps": 20,
            "beta_min": 0.1,
            "beta_max": 20.0
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "num_epochs": 100,
            "checkpoint_interval": 5,
            "visualization_frequency": 5,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "output_dir": "diffusion_ergodic/results"
        },
        "normalizer": {
            "robot_state": {
                "mean": [0.5, 0.5, 0.0, 0.0],
                "std": [0.5, 0.5, 3.14, 1.0]
            }
        }
    }
    
    # 创建配置目录
    os.makedirs("diffusion_ergodic/config", exist_ok=True)
    
    # 保存默认配置
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"已创建默认配置文件: {config_path}")
    
    # 使用load_config加载刚刚保存的配置
    return load_config(config_path)

def test_model_forward():
    """测试模型前向传播"""
    # 加载配置
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = create_default_config()
    else:
        config = load_config(config_path)
    
    print(f"使用配置文件: {config_path}")
    print(f"轨迹长度: {config.trajectory_len}")
    print(f"机器人状态维度: {config.robot_state_dim}")
    
    # 创建模型
    model = ErgodicDiffusionModel(config)
    model.train()
    
    # 创建模拟输入
    batch_size = 2
    distribution = torch.randn(batch_size, 1, 32, 32)
    robot_state = torch.randn(batch_size, config.robot_state_dim)
    trajectories = torch.randn(batch_size, config.trajectory_len, config.robot_state_dim)
    diffusion_time = torch.rand(batch_size)
    
    # 创建输入字典
    inputs = {
        'distribution': distribution,
        'robot_state': robot_state,
        'trajectories': trajectories,
        'diffusion_time': diffusion_time
    }
    
    # 前向传播
    print("执行前向传播...")
    outputs = model(inputs)
    
    # 打印输出
    print("模型输出:")
    for key, value in outputs.items():
        print(f"  {key}: 形状 {value.shape}")


def test_model_inference():
    """测试模型推理"""
    # 加载配置
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        config = create_default_config()
    else:
        config = load_config(config_path)
    
    print(f"使用配置文件: {config_path}")
    print(f"轨迹长度: {config.trajectory_len}")
    print(f"机器人状态维度: {config.robot_state_dim}")
    print(f"扩散步数: {config.diffusion.steps}")
    
    # 创建模型
    model = ErgodicDiffusionModel(config)
    model.eval()  # 设置为评估模式
    
    # 创建模拟输入
    batch_size = 2
    distribution = torch.randn(batch_size, 1, 32, 32)
    robot_state = torch.randn(batch_size, config.robot_state_dim)
    
    # 创建输入字典
    inputs = {
        'distribution': distribution,
        'robot_state': robot_state
    }
    
    # 推理
    print("执行推理...")
    try:
        outputs = model.inference(inputs)
        
        # 打印输出
        print("模型输出:")
        for key, value in outputs.items():
            print(f"  {key}: 形状 {value.shape}")
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("测试数据集加载...")
    test_dataset_loading()
    
    print("\n测试模型架构...")
    test_model_architecture()
    
    print("\n测试模型前向传播...")
    test_model_forward()
    
    print("\n测试模型推理...")
    test_model_inference()