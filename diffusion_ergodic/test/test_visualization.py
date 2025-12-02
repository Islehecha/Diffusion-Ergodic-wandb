import numpy as np
import matplotlib.pyplot as plt
from diffusion_ergodic.visualization import visualize_distribution, visualize_comparison_with_dist

# 简单测试数据
grid_size = 50
x = np.linspace(0, 3.5, grid_size)
y = np.linspace(-1, 3.5, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.exp(-((X-1.5)**2 + (Y-1.5)**2)/0.3) + 0.8*np.exp(-((X-2.5)**2 + (Y-0.5)**2)/0.2)

centers = np.array([[1.5, 1.5], [2.5, 0.5]])
workspace_bounds = np.array([[0.0, 3.5], [-1.0, 3.5]])

# 创建轨迹
t = np.linspace(0, 1, 100)
true_traj = np.column_stack([3*t, 3*t])
pred_traj = np.column_stack([3*t, 2.8*t + 0.2*np.sin(10*t)])

print("数据准备完成，开始可视化...")

# 保存而不是显示
fig, ax = visualize_distribution(
    Z, 
    workspace_bounds,
    trajectory=true_traj,
    title="测试分布",
    gaussian_centers=centers
)
plt.savefig('test_dist.png')
plt.close(fig)

print("第一个图表已保存为 test_dist.png")

fig, ax = visualize_comparison_with_dist(
    pred_traj,
    true_traj,
    Z,
    workspace_bounds,
    metric=0.123,
    title="测试轨迹比较",
    gaussian_centers=centers
)
plt.savefig('test_comparison.png')
plt.close(fig)

print("第二个图表已保存为 test_comparison.png")
print("测试完成！")