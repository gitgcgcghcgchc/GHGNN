import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

# ======== 数据准备 ========
data_mean = {
    'MSE': [118.1, 95.90, 84.58],
    'MSE_std': [10.95, 6.295, 3.376],
    'RMSE': [10.86, 9.788, 9.195],
    'RMSE_std': [0.5050, 0.3155, 0.1820],
    'MAE': [8.569, 7.580, 7.119],
    'MAE_std': [0.3527, 0.2754, 0.08605],
    'R2': [0.6744, 0.7357, 0.7668],
    'R2_std': [0.03019, 0.01735, 0.009306],
    'PCC': [0.8250, 0.8600, 0.8787],
    'PCC_std': [0.01693, 0.009643, 0.006586],
    'Spearman': [0.7648, 0.7893, 0.7946],
    'Spearman_std': [0.01171, 0.01441, 0.009673]
}

smaller_better = ['MSE', 'RMSE', 'MAE', 'MAPE']
metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'PCC', 'Spearman']
models = ['GCN', 'HGNN', 'AF-GHGNN']
colors = ['#AFCBE0', '#BCD591', '#F1BE7C']  # 浅-中-深蓝

# ======== 数据归一化 ========
ratio_means, ratio_stds, raw_means = [], [], []
for metric in metrics:
    means = np.array(data_mean[metric])
    stds = np.array(data_mean[metric + '_std'])
    raw_means.append(means)

    base = means[0]
    if metric in smaller_better:
        ratios = base / means
        rel_err = np.sqrt((stds[0] / base) ** 2 + (stds / means) ** 2)
    else:
        ratios = means / base
        rel_err = np.sqrt((stds / means) ** 2 + (stds[0] / base) ** 2)

    ratio_means.append(ratios)
    ratio_stds.append(ratios * rel_err)

ratio_means = np.array(ratio_means)
ratio_stds = np.array(ratio_stds)

# ======== 绘图设置 ========
plt.figure(figsize=(14, 7))
ax = plt.gca()
# 去除上侧和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 定义每个指标的中心位置
x = np.arange(len(metrics))  # [0, 1, 2, ...]

# 调整柱状图布局
bar_width = 0.22
group_gap = 0.1
inner_gap = 0.03

# 计算每个柱子的精确x位置
bar_positions = [x + i * (bar_width + inner_gap)
                 for i in range(len(models))]

# ======== 绘制柱状图 ========
bars_dict = {}
for i, model in enumerate(models):
    bars = ax.bar(bar_positions[i], ratio_means[:, i], width=bar_width,
                  color=colors[i], label=model, yerr=ratio_stds[:, i],
                  capsize=5, alpha=0.9)
    bars_dict[model] = bars

# ======== 绘制统一方向的折线虚线箭头 ========
# 预先计算每个指标的最大值，用于确定箭头高度
max_heights = []
for j in range(len(metrics)):
    max_height = max([bars_dict[model][j].get_height() for model in models])
    max_heights.append(max_height)


for j in range(len(metrics)):
    for i in range(len(models) - 1):
        # 获取相邻柱子
        bar1 = bars_dict[models[i]][j]
        bar2 = bars_dict[models[i + 1]][j]

        # 计算精确连接点（始终从右侧1/4出发，左侧1/4终止）
        x1 = bar1.get_x() + bar1.get_width() * 0.75  # 右侧1/4处
        x2 = bar2.get_x() + bar_width * 0.25  # 左侧1/4处
        y1 = bar1.get_height()
        y2 = bar2.get_height()

        # 强制统一箭头方向：向上出发，向下指入
        peak_y = max(y1,y2) + 0.05   # 折线最高点，添加小偏移防止重叠

        # 创建折线路径（三折线：上-中-下）
        path = [
            (x1, y1),  # 起点（右侧1/4）
            (x1, peak_y),  # 垂直向上
            (x2, peak_y),  # 水平移动
            (x2, y2)  # 垂直向下
        ]

        # 绘制虚线折线（灰色）
        for k in range(len(path) - 1):
            ax.plot([path[k][0], path[k + 1][0]], [path[k][1], path[k + 1][1]],
                    linestyle='--', linewidth=1.2, color='red', alpha=0.8)

            # ======== 在折线末端添加红色三角形 ========
            triangle_size = 0.02  # 三角形大小
            # 定义三角形顶点（向下指向）
            triangle_points = [
                (x2, y2),  # 顶点（指向下方）
                (x2 - triangle_size / 1.5, y2 + triangle_size),  # 左下角
                (x2 + triangle_size / 1.5, y2 + triangle_size)  # 右下角
            ]

            # 创建红色三角形
            triangle = Polygon(
                triangle_points,
                closed=True,
                facecolor='red',
                edgecolor='red',
                linewidth=1,
                alpha=0.9
            )
            ax.add_patch(triangle)


        # 添加百分比标签（无边框，在折线上方）
        change_pct = (y2 / y1 - 1) * 100
        label_color = '#E63946' if y2 > y1 else '#1D3557'
        ax.text(
            (x1 + x2) / 2,  # x位置居中
            peak_y + 0.03,  # y位置在折线上方
            f'{change_pct:+.1f}%',  # 带符号百分比
            ha='center',  # 水平居中
            va='bottom',  # 垂直对齐
            fontsize=8,
            color=label_color,  # 红/蓝颜色区分升降
            weight='bold',  # 加粗显示
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none')
        )
        # 获取相邻柱子
        bar1 = bars_dict[models[i]][j]
        bar2 = bars_dict[models[i + 1]][j]

        x1 = bar1.get_x() + bar1.get_width() * 0.75
        x2 = bar2.get_x() + bar_width * 0.25

        # 添加淡红色背景矩形
        rect_width = x2 - x1
        rect_height = max(y1,y2) + 0.05  # 适当高度覆盖箭头区域
        rect = plt.Rectangle((x1, 0), rect_width, rect_height,
                             facecolor='#FFCCCC', alpha=0.3, zorder=0)  # 淡红色背景
        ax.add_patch(rect)

# ======== 图表美化 ========
plt.axhline(1, color='gray', linestyle=':', linewidth=1.5)
plt.xticks(bar_positions[1] , metrics, fontsize=12)
plt.ylabel('Relative Performance (GCN=1)', fontsize=13)
plt.title('Performance Comparison: GCN vs HGNN vs AF-GHGNN Models', fontsize=14, pad=20)
plt.legend(frameon=True, fancybox=True,  fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 设置y轴范围，确保所有元素可见
plt.ylim(0.6, max(max_heights) + 0.2)

plt.tight_layout()
# 保存为高分辨率图片
plt.savefig('Performance Comparison: GCN vs HGNN vs AF-GHGNN Models.png', dpi=1000, )
plt.show()