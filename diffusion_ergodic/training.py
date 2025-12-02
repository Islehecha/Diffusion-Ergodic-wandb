import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from .visualization import visualize_distribution, visualize_comparison_with_dist
from .data_process.ergodic_processor import reconstruct_distribution
from .models.constraint_processor import PhysicsConstraintProcessor


def train_epoch(model, dataloader, optimizer, device, epoch, writer, constraint_processor=None):
    """
    简化的训练循环 - 只处理纯扩散损失
    
    Args:
        model: 扩散模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard writer
        constraint_processor: 约束处理器（用于监控，不参与训练）
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    constraint_metrics_sum = {}
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # 移动数据到设备
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        # 获取批次大小
        B = batch['distribution'].shape[0]
        
        # 使用均匀分布采样扩散时间（改进的采样策略）
        batch['diffusion_time'] = torch.rand(B, device=device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        try:
            # 前向传播
            outputs = model(batch)
            
            # 计算纯扩散损失
            loss = model.compute_loss(outputs, batch['trajectories'])
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss detected at batch {i}, skipping")
                continue
                
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            # 累加损失
            batch_loss = loss.item() * B
            running_loss += batch_loss
            total_samples += B
            
            # 记录损失组件（每10个批次记录一次）
            if writer and i % 10 == 0:
                global_step = epoch * len(dataloader) + i
                loss_components = model.get_loss_components()
                for name, value in loss_components.items():
                    writer.add_scalar(f'Training/{name}', value, global_step)
            
            # 计算约束指标用于监控（不参与训练）
            if constraint_processor and i % 20 == 0:  # 每20个批次计算一次
                with torch.no_grad():
                    pred_trajectories = outputs['prediction']
                    batch_metrics = constraint_processor.evaluate_constraints(pred_trajectories)
                    
                    # 累加约束指标
                    for name, value in batch_metrics.items():
                        if name not in constraint_metrics_sum:
                            constraint_metrics_sum[name] = []
                        constraint_metrics_sum[name].append(value)
            
        except RuntimeError as e:
            print(f"[ERROR] Runtime error at batch {i}: {e}")
            continue
    
    # 计算平均损失
    avg_loss = running_loss / total_samples if total_samples > 0 else float('nan')
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 记录约束指标（仅用于监控）
        if constraint_metrics_sum:
            for name, values in constraint_metrics_sum.items():
                avg_value = np.mean(values)
                writer.add_scalar(f'Constraint_Monitoring/{name}', avg_value, epoch)
    
    return avg_loss


def validate(model, dataloader, device, epoch=None, writer=None, sample_dir=None, 
             constraint_processor=None):
    """
    简化的验证函数
    
    Args:
        model: 扩散模型
        dataloader: 验证数据加载器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard writer
        sample_dir: 样本保存目录
        constraint_processor: 约束处理器
    """
    model.eval()
    total_loss = 0
    ergodic_metrics = []
    constraint_metrics_sum = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移至正确设备
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # 随机采样扩散时间
            B = batch['distribution'].shape[0]
            batch['diffusion_time'] = torch.rand(B, device=device)
            
            # 模型前向传播
            outputs = model(batch)
            
            # 计算损失
            loss = model.compute_loss(outputs, batch['trajectories'])
            total_loss += loss.item()
            
            # 生成轨迹并计算Ergodic指标
            pred_inputs = {
                'distribution': batch['distribution'],
                'robot_state': batch['robot_state']
            }
            
            pred_outputs = model.inference(pred_inputs)
            pred_trajectories = pred_outputs['prediction']
            
            # 计算每个样本的Ergodic指标（简化为MSE）
            for i in range(B):
                metric = nn.MSELoss()(pred_trajectories[i], batch['trajectories'][i])
                ergodic_metrics.append(metric.item())
            
            # 计算约束指标
            if constraint_processor:
                batch_metrics = constraint_processor.evaluate_constraints(pred_trajectories)
                for name, value in batch_metrics.items():
                    if name not in constraint_metrics_sum:
                        constraint_metrics_sum[name] = []
                    constraint_metrics_sum[name].append(value)
            
            # 生成可视化（仅对第一个批次）
            if sample_dir is not None and batch_idx < 1:
                _generate_validation_visualizations(
                    batch, pred_trajectories, sample_dir, epoch, batch_idx, constraint_processor
                )
    
    avg_loss = total_loss / len(dataloader)
    avg_ergodic_metric = np.mean(ergodic_metrics) if ergodic_metrics else 0.0
    
    # 记录到TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/validation', avg_loss, epoch)
        writer.add_scalar('Metrics/ergodic', avg_ergodic_metric, epoch)
        
        # 记录约束指标
        if constraint_metrics_sum:
            for name, values in constraint_metrics_sum.items():
                avg_value = np.mean(values)
                writer.add_scalar(f'Validation_Constraints/{name}', avg_value, epoch)
    
    return avg_loss, avg_ergodic_metric


def _generate_validation_visualizations(batch, pred_trajectories, sample_dir, epoch, batch_idx, 
                                      constraint_processor=None):
    """生成验证可视化"""
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(min(1, batch['trajectories'].shape[0])):  # 只处理第一个样本
        pred_traj = pred_trajectories[i].cpu().numpy()
        true_traj = batch['trajectories'][i].cpu().numpy()
        
        # 计算metric
        metric = ((pred_traj - true_traj) ** 2).mean()
        
        # 重建分布（简化版本）
        try:
            orig_dist = batch['distribution'][i, 0].cpu().numpy()
            if len(orig_dist.shape) == 1:
                grid_size = int(np.sqrt(len(orig_dist)))
                if grid_size * grid_size == len(orig_dist):
                    orig_dist = orig_dist.reshape(grid_size, grid_size)
                else:
                    orig_dist = np.ones((32, 32)) * 0.1  # 默认网格
        except:
            orig_dist = np.ones((32, 32)) * 0.1  # 默认网格
        
        # 工作空间边界
        workspace_bounds = np.array([[0.0, 3.5], [-1.0, 3.5]])
        
        # 生成比较图像
        sample_id = f"val_e{epoch}_b{batch_idx}_s{i}"
        comp_path = os.path.join(sample_dir, f'comparison_{sample_id}.png')
        
        try:
            fig, ax = visualize_comparison_with_dist(
                pred_traj,
                true_traj,
                orig_dist,
                workspace_bounds,
                metric=metric,
                title=f"Trajectory Comparison (Epoch {epoch})"
            )
            plt.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"保存轨迹比较图像到 {comp_path}")
        except Exception as e:
            print(f"生成可视化时出错: {e}")
        
        # 如果有约束处理器，生成约束分析
        if constraint_processor:
            try:
                # 应用约束后的轨迹
                pred_traj_tensor = torch.tensor(pred_traj).unsqueeze(0).float()
                constrained_traj = constraint_processor.apply_constraints(pred_traj_tensor, 0.5)
                constrained_traj_np = constrained_traj[0].numpy()
                
                # 保存约束后的比较
                constrained_path = os.path.join(sample_dir, f'constrained_{sample_id}.png')
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # 原始预测
                axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Original Prediction')
                axes[0].plot(true_traj[:, 0], true_traj[:, 1], 'r--', label='Ground Truth')
                axes[0].set_title('Original Prediction')
                axes[0].legend()
                axes[0].grid(True)
                
                # 约束后预测
                axes[1].plot(constrained_traj_np[:, 0], constrained_traj_np[:, 1], 'g-', label='Constrained')
                axes[1].plot(true_traj[:, 0], true_traj[:, 1], 'r--', label='Ground Truth')
                axes[1].set_title('After Constraints')
                axes[1].legend()
                axes[1].grid(True)
                
                plt.tight_layout()
                plt.savefig(constrained_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"生成约束可视化时出错: {e}")


def create_constraint_processor(config):
    """创建约束处理器的辅助函数"""
    if hasattr(config, 'constraints'):
        return PhysicsConstraintProcessor(config.constraints)
    return None