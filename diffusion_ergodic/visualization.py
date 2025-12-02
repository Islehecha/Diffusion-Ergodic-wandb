import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from .data_process.ergodic_processor import reconstruct_distribution

def visualize_distribution(distribution, workspace_bounds, save_path=None, trajectory=None, metric=None, title=None, gaussian_centers=None):
    """可视化原始分布，使用等高线图显示高斯分布的三维形状"""
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格
    x_min, x_max = workspace_bounds[0]
    y_min, y_max = workspace_bounds[1]
    
    # 计算网格尺寸（保持与分布形状匹配）
    grid_shape = distribution.shape
    
    # 创建网格坐标
    x = np.linspace(x_min, x_max, grid_shape[1])
    y = np.linspace(y_min, y_max, grid_shape[0])
    X, Y = np.meshgrid(x, y)
    
    # 正规化分布值以增强对比度
    Z = distribution / np.max(distribution)
    
    # 使用紫色背景
    ax.set_facecolor('#4c2a7c')  # 设置接近图2的紫色背景色
    
    # 绘制等高线填充 - 使用viridis颜色映射并适当调整透明度
    # 使用更多的级别使过渡更平滑
    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.8)
    
    # 绘制等高线 - 使用较多的等高线级别以显示更细腻的变化
    # 关键点：这里使用黑色等高线来表示3D表面
    contour_lines = ax.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    # 如果有轨迹，则绘制轨迹（仅截掉尾部零填充，避免跨段直线伪影）
    if trajectory is not None:
        traj_np = trajectory
        if isinstance(traj_np, torch.Tensor):
            traj_np = traj_np.cpu().numpy()
        # 使用“整行全为零”判断填充（避免将合法 (0,0) 坐标误判为填充）
        nonzero_rows = ~np.all(traj_np == 0, axis=1)
        valid_len = np.where(nonzero_rows)[0].max() + 1 if nonzero_rows.any() else len(traj_np)
        valid_traj = traj_np[:valid_len]
        ax.plot(valid_traj[:, 0], valid_traj[:, 1], 'r-', linewidth=2)
        ax.plot(valid_traj[0, 0], valid_traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(valid_traj[-1, 0], valid_traj[-1, 1], 'bo', markersize=8, label='End')
        ax.legend()
    
    # 设置标题
    if title is None:
        if gaussian_centers is not None and isinstance(gaussian_centers, np.ndarray):
            title = f"Trajectory dist_{0:04d}_traj_{0:04d}"
        else:
            title = "Distribution"
    if metric is not None:
        title += f", Ergodic Metric: {metric:.6f}"
        if hasattr(trajectory, 'shape') and trajectory.shape[0] > 1:
            # 估算时间，假设每个步骤0.1秒
            traj_time = trajectory.shape[0] * 0.1
            title += f", Time: {traj_time:.2f}s"
    ax.set_title(title)
    
    # 设置坐标轴
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(workspace_bounds[0])
    ax.set_ylim(workspace_bounds[1])
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
        
    return fig, ax

def visualize_comparison_with_dist(pred_traj, true_traj, distribution, workspace_bounds, save_path=None, metric=None, title=None, gaussian_centers=None):
    """可视化预测轨迹和真实轨迹的对比，背景为原始分布的等高线图"""
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格
    x_min, x_max = workspace_bounds[0]
    y_min, y_max = workspace_bounds[1]
    
    # 计算网格尺寸（保持与分布形状匹配）
    grid_shape = distribution.shape
    
    # 创建网格坐标
    x = np.linspace(x_min, x_max, grid_shape[1])
    y = np.linspace(y_min, y_max, grid_shape[0])
    X, Y = np.meshgrid(x, y)
    
    # 正规化分布值以增强对比度
    Z = distribution / np.max(distribution)
    
    # 设置紫色背景色
    ax.set_facecolor('#4c2a7c')  # 深紫色背景
    
    # 使用viridis色图绘制等高线填充
    # 关键：增加levels数量以获得更平滑的渐变
    contour = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    
    # 绘制黑色等高线，增加数量使形状更清晰
    # 这是关键部分 - 增加等高线显示三维形状
    contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # 绘制中心点
    if gaussian_centers is not None and len(gaussian_centers) > 0:
        centers = np.array(gaussian_centers)
        ax.scatter(centers[:, 0], centers[:, 1], c='r', s=80, marker='o', label='Gaussian Centers')
        
        # 为每个中心点添加编号
        for i, center in enumerate(centers):
            ax.text(center[0], center[1], f"{i+1}", color='white', 
                   fontsize=10, ha='center', va='center')
    
    # 绘制真实轨迹：仅截掉尾部零填充（避免跨段直线伪影）
    gt_np = true_traj
    if isinstance(gt_np, torch.Tensor):
        gt_np = gt_np.cpu().numpy()
    gt_nonzero = ~np.all(gt_np == 0, axis=1)
    gt_valid_len = np.where(gt_nonzero)[0].max() + 1 if gt_nonzero.any() else len(gt_np)
    gt_valid = gt_np[:gt_valid_len]
    ax.plot(gt_valid[:, 0], gt_valid[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(gt_valid[0, 0], gt_valid[0, 1], 'go', markersize=8, label='Start')
    ax.plot(gt_valid[-1, 0], gt_valid[-1, 1], 'mo', markersize=8, label='End (GT)')

    # 绘制预测轨迹：仅截掉尾部零填充
    pred_np = pred_traj
    if isinstance(pred_np, torch.Tensor):
        pred_np = pred_np.cpu().numpy()
    pred_nonzero = ~np.all(pred_np == 0, axis=1)
    pred_valid_len = np.where(pred_nonzero)[0].max() + 1 if pred_nonzero.any() else len(pred_np)
    pred_valid = pred_np[:pred_valid_len]
    ax.plot(pred_valid[:, 0], pred_valid[:, 1], 'r-', linewidth=2, label='Predicted')
    ax.plot(pred_valid[-1, 0], pred_valid[-1, 1], 'bo', markersize=8, label='End (Pred)')
    
    # 设置标题
    if title is None:
        title = "Trajectory Comparison"
    if metric is not None:
        title += f", Metric: {metric:.6f}"
    ax.set_title(title)
    
    # 设置坐标轴
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(workspace_bounds[0])
    ax.set_ylim(workspace_bounds[1])
    
    # 不添加颜色条，这样更接近目标图像
    # plt.colorbar(contour, ax=ax)
    
    ax.legend()
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
        
    return fig, ax

def visualize_trajectories(model, data_loader, device, epoch=None, writer=None, n_samples=3):
    """
    生成并可视化一些轨迹样本，使用等高线图显示分布
    """
    # 如果不需要可视化，直接返回
    if writer is None:
        return
        
    model.eval()
    
    # 获取一个批次的数据
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    # 将数据移到设备
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)
    
    # 从批次中选择n_samples个样本
    subset = {}
    for key in batch:
        if torch.is_tensor(batch[key]):
            subset[key] = batch[key][:n_samples]
        else:
            subset[key] = batch[key][:n_samples]
    
    # 生成轨迹
    with torch.no_grad():
        try:
            # A.尝试使用模型的标准推理方法
            outputs = model.inference(subset)
            if 'prediction' in outputs:
                generated_trajectories = outputs['prediction']
            elif 'trajectories' in outputs:
                generated_trajectories = outputs['trajectories']
            else:
                print("警告: 无法从模型输出中找到轨迹数据")
                return
        except AttributeError:
            try:
                # B.如果模型有编码器和解码器
                encoder_outputs = model.encoder(subset)
                outputs = model.decoder.inference(encoder_outputs, subset)
                generated_trajectories = outputs['trajectories']
            except AttributeError as e:
                print(f"错误: 无法生成轨迹: {e}")
                return
    
    # 创建可视化图
    fig = plt.figure(figsize=(15, 4 * n_samples))
    
    # 使用统一的工作空间边界
    workspace_bounds = np.array([[0.0, 3.5], [-1.0, 3.5]])
    
    # 对每个样本进行可视化
    for i in range(n_samples):
        # 获取高斯中心点（如果存在）
        centers = None
        if 'gaussian_params' in subset and 'centers' in subset['gaussian_params']:
            try:
                centers = subset['gaussian_params']['centers'][i]
                if not isinstance(centers, np.ndarray):
                    centers = centers.cpu().numpy()
                # 获取实际的高斯数量
                n_gaussians = subset['gaussian_params']['n_gaussians']
                if isinstance(n_gaussians, list):
                    centers = centers[:n_gaussians[i]]
                elif hasattr(n_gaussians, 'item'):
                    centers = centers[:n_gaussians.item()]
                elif isinstance(n_gaussians, int):
                    centers = centers[:n_gaussians]
            except (IndexError, TypeError, AttributeError):
                pass
        
        # 创建原始分布网格
        dist = subset['distribution'][i, 0].cpu().numpy()
        
        # 正规化分布值以增强对比度
        dist = dist / np.max(dist)
        
        # 计算网格
        x_min, x_max = workspace_bounds[0]
        y_min, y_max = workspace_bounds[1]
        x = np.linspace(x_min, x_max, dist.shape[1])
        y = np.linspace(y_min, y_max, dist.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # 子图1: 原始分布
        ax1 = fig.add_subplot(n_samples, 3, i * 3 + 1)
        ax1.set_facecolor('#4c2a7c')  # 设置紫色背景
        contour = ax1.contourf(X, Y, dist, levels=15, cmap='viridis')
        contour_lines = ax1.contour(X, Y, dist, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        
        # 如果有中心点，显示它们
        if centers is not None and len(centers) > 0:
            for j, center in enumerate(centers):
                ax1.scatter(center[0], center[1], c='r', s=80, marker='o')
                ax1.text(center[0], center[1], f"{j+1}", color='white', 
                       fontsize=9, ha='center', va='center')
        
        ax1.set_title(f"Sample {i+1} - Distribution")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_xlim(workspace_bounds[0])
        ax1.set_ylim(workspace_bounds[1])
        
        # 子图2: 真实轨迹
        ax2 = fig.add_subplot(n_samples, 3, i * 3 + 2)
        ax2.set_facecolor('#4c2a7c')  # 设置紫色背景
        true_traj = subset['trajectories'][i].cpu().numpy()
        
        # 绘制分布背景
        contour2 = ax2.contourf(X, Y, dist, levels=15, cmap='viridis')
        contour_lines2 = ax2.contour(X, Y, dist, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        
        # 绘制轨迹（仅截掉尾部零填充）
        nz2 = ~np.all(true_traj == 0, axis=1)
        vl2 = np.where(nz2)[0].max() + 1 if nz2.any() else len(true_traj)
        true_valid = true_traj[:vl2]
        ax2.plot(true_valid[:, 0], true_valid[:, 1], 'r-', linewidth=2, label='True')
        ax2.plot(true_valid[0, 0], true_valid[0, 1], 'go', markersize=8, label='Start')
        ax2.plot(true_valid[-1, 0], true_valid[-1, 1], 'bo', markersize=8, label='End')
        ax2.set_title(f"Sample {i+1} - True Trajectory")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim(workspace_bounds[0])
        ax2.set_ylim(workspace_bounds[1])
        ax2.legend()
        
        # 子图3: 生成轨迹
        ax3 = fig.add_subplot(n_samples, 3, i * 3 + 3)
        ax3.set_facecolor('#4c2a7c')  # 设置紫色背景
        gen_traj = generated_trajectories[i].cpu().numpy()
        
        # 绘制分布背景
        contour3 = ax3.contourf(X, Y, dist, levels=15, cmap='viridis')
        contour_lines3 = ax3.contour(X, Y, dist, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        
        # 绘制轨迹（仅截掉尾部零填充）
        nz3 = ~np.all(gen_traj == 0, axis=1)
        vl3 = np.where(nz3)[0].max() + 1 if nz3.any() else len(gen_traj)
        gen_valid = gen_traj[:vl3]
        ax3.plot(gen_valid[:, 0], gen_valid[:, 1], 'r-', linewidth=2, label='Generated')
        ax3.plot(gen_valid[0, 0], gen_valid[0, 1], 'go', markersize=8, label='Start')
        ax3.plot(gen_valid[-1, 0], gen_valid[-1, 1], 'bo', markersize=8, label='End')
        ax3.set_title(f"Sample {i+1} - Generated Trajectory")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_xlim(workspace_bounds[0])
        ax3.set_ylim(workspace_bounds[1])
        ax3.legend()
    
    plt.tight_layout()
    
    # 添加图到TensorBoard
    if writer and epoch is not None:
        writer.add_figure('Trajectories', fig, epoch)
    
    # 清理
    plt.close(fig)
    
def generate_samples(model, dataloader, device, num_samples=5, save_dir=None):
    """生成样本并可视化"""
    if save_dir is None:
        # 设置默认保存目录到项目根目录下的visualizations文件夹
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        # 获取一批数据
        batch = next(iter(dataloader))
        
        # 将数据移至正确设备
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # 只处理请求的样本数量
        B = min(num_samples, batch['distribution'].shape[0])
        
        # 生成预测轨迹
        pred_inputs = {
            'distribution': batch['distribution'][:B],
            'robot_state': batch['robot_state'][:B]
        }
        
        pred_outputs = model.inference(pred_inputs)
        
        if 'prediction' in pred_outputs:
            pred_trajectories = pred_outputs['prediction']
        elif 'trajectories' in pred_outputs:
            pred_trajectories = pred_outputs['trajectories']
        else:
            print(f"警告: 无法从推理输出中找到轨迹数据, 可用键: {pred_outputs.keys()}")
            return
        
        # 为每个样本生成可视化
        for i in range(B):
            # 获取轨迹数据
            pred_traj = pred_trajectories[i].cpu().numpy()
            true_traj = batch['trajectories'][i].cpu().numpy()
            
            # 计算metric
            metric = ((pred_traj - true_traj) ** 2).mean()
            
            # 获取高斯参数
            centers = None
            covs = None
            weights = None
            n_gaussians = None
            
            if 'gaussian_params' in batch:
                if 'centers' in batch['gaussian_params']:
                    centers = batch['gaussian_params']['centers'][i].cpu().numpy()
                if 'covs' in batch['gaussian_params']:
                    covs = batch['gaussian_params']['covs'][i].cpu().numpy()
                if 'weights' in batch['gaussian_params']:
                    weights = batch['gaussian_params']['weights'][i].cpu().numpy()
                if 'n_gaussians' in batch['gaussian_params']:
                    # 修复这一行: 针对单个样本提取 n_gaussians
                    if torch.is_tensor(batch['gaussian_params']['n_gaussians']):
                        # 如果是张量，提取第 i 个元素
                        n_gaussians = batch['gaussian_params']['n_gaussians'][i].item()
                    else:
                        # 如果是列表，直接索引
                        n_gaussians = batch['gaussian_params']['n_gaussians'][i]
            
            # 使用统一的工作空间边界
            workspace_bounds = np.array([[0.0, 3.5], [-1.0, 3.5]])
            
            # 重建分布网格
            if centers is not None and covs is not None and weights is not None and n_gaussians is not None:
                # 检查 n_gaussians 是否是有效值
                if isinstance(n_gaussians, (int, float)) and n_gaussians > 0:
                    # 仅使用实际高斯组件数量
                    valid_centers = centers[:n_gaussians]
                    valid_covs = covs[:n_gaussians]
                    valid_weights = weights[:n_gaussians]
                    
                    # 重建分布网格
                    # print(f"使用 {n_gaussians} 个高斯组件重建分布网格")
                    orig_dist = reconstruct_distribution(
                        valid_centers, valid_covs, valid_weights, workspace_bounds, grid_size=50
                    )
                    # print(f"重建的分布网格形状: {orig_dist.shape}")
                else:
                    print(f"警告: n_gaussians 值无效: {n_gaussians}，使用原始分布")
                    orig_dist = batch['distribution'][i, 0].cpu().numpy()
            else:
                # 如果无法从高斯参数重建，尝试使用原始分布
                print("无法从高斯参数重建分布，使用原始分布")
                orig_dist = batch['distribution'][i, 0].cpu().numpy()
                
                # 如果原始分布是平坦的数组，尝试重塑为2D网格
                if len(orig_dist.shape) == 1:
                    # 尝试自动检测正方形网格大小
                    grid_size = int(np.sqrt(len(orig_dist)))
                    if grid_size * grid_size == len(orig_dist):
                        print(f"将原始分布重塑为 {grid_size}x{grid_size} 网格")
                        orig_dist = orig_dist.reshape(grid_size, grid_size)
            
            # 检查分布是否为2D网格
            if len(orig_dist.shape) != 2:
                print(f"警告: 分布不是2D网格，形状: {orig_dist.shape}")
                # 创建一个空的2D网格
                orig_dist = np.ones((50, 50)) * 0.1
            
            # 确定要用于可视化的中心点
            vis_centers = None
            if centers is not None and n_gaussians is not None and isinstance(n_gaussians, (int, float)) and n_gaussians > 0:
                vis_centers = centers[:n_gaussians]
            
            # 生成和保存可视化
            sample_path = os.path.join(save_dir, f'sample_{i+1}.png')
            try:
                fig, ax = visualize_comparison_with_dist(
                    pred_traj,
                    true_traj,
                    orig_dist,
                    workspace_bounds,
                    metric=metric,
                    title=f"Sample {i+1}",
                    gaussian_centers=vis_centers
                )
                plt.savefig(sample_path)
                plt.close(fig)
                print(f"已保存样本 {i+1} 到 {sample_path}")
            except Exception as e:
                print(f"生成可视化时出错: {e}")
