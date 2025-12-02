Diffusion-Ergodic
├─ README.md % 项目根目录的说明文件
├─ deployment % 部署平台相关代码，独立于主项目
│  ├─ README.md % 部署模块的说明文件
│  ├─ diffusion_model_interface.py % 部署平台的模型推理接口，封装了模型加载和调用逻辑
│  ├─ dpm_sampler.py % 部署端使用的DPM采样器实现
│  ├─ sde.py % 部署端使用的随机微分方程(SDE)定义
│  └─ test_interface.py % 用于测试部署接口功能的脚本
├─ diffusion_ergodic % 项目核心代码
│  ├─ config
│  │  └─ config_ergodic.yaml % 主配置文件，包含所有模型、数据和训练参数
│  ├─ data
│  │  └─ ergodic_dataset % 存放训练和验证用的数据集
│  ├─ data_process
│  │  ├─ __init__.py
│  │  └─ ergodic_processor.py % 数据处理模块，定义Dataset和DataLoader
│  ├─ evaluate.py % 评估模型性能的脚本
│  ├─ main.py % 项目主入口，负责启动训练、验证流程
│  ├─ models % 存放所有模型架构相关代码
│  │  ├─ __init__.py
│  │  ├─ constraint_processor.py % 物理约束后处理器，用于优化轨迹
│  │  ├─ diffusion_ergodic.py % 定义核心的遍历扩散模型(ErgodicDiffusionModel)
│  │  ├─ diffusion_utils % 扩散模型相关的工具函数
│  │  │  ├─ __init__.py
│  │  │  ├─ dpm_solver_pytorch.py % DPM采样器核心算法
│  │  │  ├─ sampling.py % 封装各种采样策略的脚本
│  │  │  └─ sde.py % 训练端使用的随机微分方程(SDE)定义
│  │  └─ module % 模型的核心组件
│  │     ├─ __init__.py
│  │     ├─ decoder.py % 解码器模块，基于DiT生成轨迹
│  │     ├─ dit.py % Diffusion Transformer (DiT)核心实现
│  │     └─ encoder.py % 编码器模块，处理分布图和初始状态
│  ├─ requirements.txt % 项目所需的Python依赖包列表
│  ├─ trained % 存放训练好的模型权重和相关配置
│  │  ├─ best_model.pth % 训练过程中保存的最佳模型权重文件
│  │  └─ model_config.yaml % 与最佳模型匹配的配置文件快照
│  ├─ training.py % 定义训练循环(training loop)的逻辑
│  └─ visualization.py % 可视化工具，用于生成轨迹对比图等
└─ weekly.md % 周报或项目进展记录文件

描述 Diffusion-Ergodic 项目核心工作流程的流程图
graph TD
    subgraph "用户操作"
        A[执行: python3 diffusion_ergodic/main.py] --> B{main.py};
    end

    subgraph "main.py: 项目总指挥"
        B --> C[1. 加载配置文件<br>(config_ergodic.yaml)];
        C --> D[2. 初始化<br>- 创建模型 (ErgodicDiffusionModel)<br>- 创建数据加载器 (get_data_loaders)<br>- 创建优化器和学习率调度器<br>- 创建日志/模型保存目录];
        D --> E{For each epoch in num_epochs};
    end

    subgraph "training.py: 训练执行者"
        E -- "开始训练" --> F[train_epoch];
        F --> G{For each batch in train_loader};
        G --> H[前向传播<br>计算扩散损失];
        H --> I[反向传播<br>更新模型权重];
        I --> G;
        G -- "Epoch 训练结束" --> J[计算平均训练损失];
        J --> E;
    end

    subgraph "training.py: 验证执行者"
        E -- "达到验证频率" --> K[validate];
        K --> L{For each batch in val_loader};
        L --> M[模型推理<br>生成预测轨迹];
        M --> N[计算验证损失<br>和遍历度量];
        N --> L;
        L -- "Epoch 验证结束" --> O[计算平均验证指标];
    end
    
    subgraph "visualization.py: 可视化模块"
        M -- "需要可视化" --> P[generate_samples<br>生成轨迹对比图];
        P --> K;
    end

    subgraph "main.py: 决策与管理"
        O --> Q{与历史最佳模型比较};
        Q -- "是更优模型" --> R[保存 best_model.pth];
        Q -- "否则" --> E;
        R --> E;
        E -- "训练循环结束" --> S[结束训练];
    end

    style A fill:#cde4ff,stroke:#333,stroke-width:2px
    style S fill:#cde4ff,stroke:#333,stroke-width:2px

——————————————————————————————————————————————————————————
流程图解读
1. 启动: 用户从命令行执行 main.py，启动整个流程。
2. 初始化 (main.py):
main.py 首先扮演“项目经理”的角色，读取配置文件，并准备好所有需要的资源：模型、数据、优化器、文件夹等。
3. 主训练循环 (main.py):
进入一个大的循环，按 epoch 进行迭代。
4. 训练阶段 (training.py):
在每个 epoch 中，main.py 调用 train_epoch 函数。
train_epoch 负责具体的训练细节，它会遍历训练数据加载器中的每一个批次，执行“前向传播 -> 计算损失 -> 反向传播 -> 更新权重”这个核心步骤。
完成一个 epoch 的训练后，它将平均损失汇报给 main.py。
5. 验证阶段 (training.py):
当达到预设的验证频率时（例如每5个 epoch），main.py 会调用 validate 函数。
validate 函数在验证集上对模型进行评估，计算损失和关键性能指标（如遍历度量）。
如果需要，它还会调用可视化模块 (visualization.py) 来生成图像，以便直观地检查模型效果。
6. 决策与保存 (main.py):
validate 函数将验证结果汇报给 main.py。
main.py 根据验证结果进行决策：如果当前模型的性能超过了历史最佳，就将当前模型权重保存为 best_model.pth。
7. 结束:
当所有 epoch 都完成后，训练循环结束。