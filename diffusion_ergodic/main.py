import os
import sys
import yaml
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_ergodic.models.diffusion_ergodic import ErgodicDiffusionModel
from diffusion_ergodic.data_process.ergodic_processor import get_data_loaders
from diffusion_ergodic.visualization import generate_samples
from diffusion_ergodic.training import train_epoch, validate, create_constraint_processor


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 转换为对象以便属性访问
    class Config:
        def __init__(self, dic):
            for key, value in dic.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)

        def __repr__(self):
            attrs = ', '.join([f"{k}={v}" for k, v in self.__dict__.items()])
            return f"Config({attrs})"

    config = Config(config_dict)

    # 为了兼容模型代码，添加一些顶层属性
    if hasattr(config, 'diffusion'):
        config.beta_min = config.diffusion.beta_min
        config.beta_max = config.diffusion.beta_max
        config.diffusion_model_type = config.diffusion.model_type
        config.diffusion_steps = config.diffusion.steps
        config.start_point_coef = config.diffusion.start_point_coef

    # 数据参数
    if hasattr(config, 'data'):
        config.data_dir = config.data.data_dir
        config.trajectory_len = config.data.trajectory_len
        config.robot_state_dim = config.data.robot_state_dim
        config.distribution_dim = config.data.distribution_dim
        config.validation_split = config.data.validation_split
        config.shuffle_dataset = config.data.shuffle_dataset
        config.num_workers = config.data.num_workers
        config.seed = config.data.seed

    # 模型参数
    if hasattr(config, 'model'):
        config.hidden_dim = config.model.hidden_dim
        config.encoder_depth = config.model.encoder_depth
        config.decoder_depth = config.model.decoder_depth
        config.num_heads = config.model.num_heads
        config.encoder_drop_path_rate = config.model.encoder_drop_path_rate
        config.decoder_drop_path_rate = config.model.decoder_drop_path_rate

    # 训练参数
    if hasattr(config, 'training'):
        config.batch_size = config.training.batch_size
        config.learning_rate = config.training.learning_rate
        config.weight_decay = config.training.weight_decay
        config.num_epochs = config.training.num_epochs
        config.device = config.training.device
        config.output_dir = config.training.output_dir

    # 处理normalizer
    if hasattr(config, 'normalizer') and hasattr(config.normalizer, 'robot_state'):
        config.normalizer.robot_state.mean = torch.tensor(config.normalizer.robot_state.mean)
        config.normalizer.robot_state.std = torch.tensor(config.normalizer.robot_state.std)

    return config


def config_to_dict(config_obj):
    """递归地将Config对象转换为纯字典"""
    if not hasattr(config_obj, '__dict__'):
        if isinstance(config_obj, torch.Tensor):
            return config_obj.tolist()
        return config_obj

    result = {}
    for key, value in config_obj.__dict__.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            result[key] = config_to_dict(value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description='简化的遍历扩散模型训练')
    parser.add_argument('--config', type=str, default='diffusion_ergodic/config/config_ergodic.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--debug', action='store_true', help='调试模式，不保存模型、图片和日志')
    parser.add_argument('--constraint_strength', type=float, default=None,
                       help='约束强度 (0.0-1.0)，覆盖配置文件设置')
    # Ultra dataset toggles (默认指向标准小规模数据集，便于快速验证)
    parser.add_argument('--use_ultra_dataset', action='store_true',
                           help='使用超大/指定数据集覆盖配置中的数据路径与轨迹长度')
    parser.add_argument('--ultra_data_dir', type=str, default='/home/songxy/code/time-optimal-ergodic-search/experiments/bias_search/ergodic_dataset',
                           help='当 --use_ultra_dataset 指定时使用的数据集目录（默认指向标准数据集）')
    parser.add_argument('--ultra_traj_len', type=int, default=101,
                           help='当 --use_ultra_dataset 指定时使用的轨迹长度（标准数据集默认 101）')

    args = parser.parse_args()

    # 先加载配置
    config = load_config(args.config)

    # 再根据开关覆盖数据集参数（可选）
    if getattr(args, 'use_ultra_dataset', False):
        if hasattr(config, 'data'):
            config.data.data_dir = args.ultra_data_dir
            config.data.trajectory_len = args.ultra_traj_len
        config.data_dir = args.ultra_data_dir
        config.trajectory_len = args.ultra_traj_len
        print(f"使用覆盖数据集: data_dir={config.data_dir}, trajectory_len={config.trajectory_len}")

    # 标记是否为调试模式
    debug_mode = args.debug
    if debug_mode:
        print("【调试模式】已启用 - 将不保存模型、图片和日志")

    # 覆盖约束强度
    if args.constraint_strength is not None:
        if hasattr(config, 'constraints'):
            config.constraints.default_constraint_strength = args.constraint_strength
        print(f"从命令行覆盖约束强度: {args.constraint_strength}")

    # 设置设备
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader = get_data_loaders(config)
    print(f"训练集大小: {len(train_loader.sampler)}, 验证集大小: {len(val_loader.sampler)}")

    # 创建模型
    model = ErgodicDiffusionModel(config).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"模型类型: {model.model_type}")

    # 创建约束处理器
    constraint_processor = create_constraint_processor(config)
    if constraint_processor:
        print(f"约束处理器已创建，默认强度: {constraint_processor.default_constraint_strength}")

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 创建运行目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ergodic_diffusion_clean_{timestamp}"

    # 在调试模式下使用临时目录
    if debug_mode:
        save_dir = os.path.join('/tmp', run_name)
        log_dir = os.path.join('/tmp', 'runs', run_name)
    else:
        save_dir = os.path.join(config.training.output_dir, run_name)
        log_dir = os.path.join('runs', run_name)

    # 创建TensorBoard writer
    writer = None
    if not debug_mode:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志目录: {log_dir}")

        # 记录训练参数
        writer.add_text('Parameters/model_type', config.diffusion.model_type, 0)
        writer.add_text('Parameters/batch_size', str(config.training.batch_size), 0)
        writer.add_text('Parameters/learning_rate', str(config.training.learning_rate), 0)
        writer.add_text('Parameters/num_epochs', str(config.training.num_epochs), 0)
        if constraint_processor:
            writer.add_text('Parameters/constraint_strength',
                          str(constraint_processor.default_constraint_strength), 0)
    else:
        # 调试模式下的dummy writer
        class DummyWriter:
            def add_scalar(self, *args, **kwargs): pass
            def add_text(self, *args, **kwargs): pass
            def add_figure(self, *args, **kwargs): pass
            def close(self): pass
        writer = DummyWriter()
        print("【调试模式】TensorBoard日志记录已禁用")

    # 恢复训练
    start_epoch = 0
    if args.resume and not debug_mode:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"检查点已加载。将从epoch {start_epoch}继续训练。")
        else:
            print(f"未找到检查点 '{args.resume}'")

    # 创建保存目录并保存配置
    if not debug_mode:
        os.makedirs(save_dir, exist_ok=True)

        # 保存配置
        config_dict = config_to_dict(config)
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # 保存到trained目录
        os.makedirs('diffusion_ergodic/trained', exist_ok=True)
        with open(os.path.join('diffusion_ergodic/trained', 'model_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        print("【调试模式】配置文件保存已禁用")

    # 训练循环
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = getattr(config.training, 'early_stopping_patience', 15)

    # 在调试模式下减少训练周期
    if debug_mode:
        original_epochs = config.training.num_epochs
        config.training.num_epochs = min(3, original_epochs)
        print(f"【调试模式】训练周期已限制为 {config.training.num_epochs} (原始值: {original_epochs})")

    print(f"开始训练，共 {config.training.num_epochs} 个epoch")

    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"\\n=== Epoch {epoch}/{config.training.num_epochs-1} ===")

        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, constraint_processor
        )

        # 确定是否需要进行可视化
        do_visualization = (
            not debug_mode and (
                epoch % max(1, config.training.visualization_frequency) == 0 or
                epoch == config.training.num_epochs-1
            )
        )

        # 验证
        sample_dir = None
        if do_visualization:
            sample_dir = os.path.join(save_dir, f'visualizations_epoch_{epoch}')
            os.makedirs(sample_dir, exist_ok=True)

        val_loss, ergodic_metric = validate(
            model, val_loader, device, epoch, writer, sample_dir, constraint_processor
        )

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if not debug_mode:
            writer.add_scalar('Learning_rate', current_lr, epoch)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印信息
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Ergodic Metric: {ergodic_metric:.6f}, LR: {current_lr:.6f}")

        # 保存模型
        if not debug_mode:
            is_best = val_loss < best_val_loss

            # 定期保存或最佳模型
            if epoch % config.training.checkpoint_interval == 0 or is_best or epoch == config.training.num_epochs-1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'ergodic_metric': ergodic_metric,
                    'config': config_dict
                }

                # 定期保存
                if epoch % config.training.checkpoint_interval == 0 or epoch == config.training.num_epochs-1:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"保存检查点到 {checkpoint_path}")

                # 最佳模型保存
                if is_best:
                    best_model_path = os.path.join('diffusion_ergodic/trained', 'best_model.pth')
                    torch.save(checkpoint, best_model_path)

                    run_best_model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(checkpoint, run_best_model_path)

                    print(f"保存最佳模型到 {best_model_path}，验证损失: {val_loss:.6f}")
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"验证损失 {patience} 个epoch未改善，早停在 epoch {epoch}")
                        break
        else:
            print("【调试模式】模型保存已禁用")

        # 生成详细样本可视化
        if do_visualization and epoch % config.training.visualization_frequency == 0:
            generate_samples(model, val_loader, device, num_samples=3, save_dir=sample_dir)
            print(f"已生成可视化结果保存至: {sample_dir}")

    # 关闭TensorBoard writer
    writer.close()

    if not debug_mode:
        print(f"\\n训练完成! TensorBoard 日志保存在 {log_dir}")
        print(f"可以使用命令启动 TensorBoard 查看结果: tensorboard --logdir={os.path.dirname(log_dir)}")
        print(f"模型保存在: {save_dir}")
    else:
        print("【调试模式】训练完成，临时文件未保存")


if __name__ == '__main__':
    main()