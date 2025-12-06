import matplotlib.pyplot as plt
import numpy as np

from graph_emb import load_pre,load_pre1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# 设置Seaborn主题和调色板
sns.set_style("whitegrid")
palette = sns.color_palette("pastel")
# 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 你可以根据需要调整字体大小

# 假设我们有一些训练集和测试集的数据

file_fold = f'../save_result/StructΔTm801/GHG/PoincareBall/2025_10_09_15_47/'
# file_fold = f'../save_result/real/GCN/Hyperboloid/2025_08_20_12_07/'
pre_couple_file_train = file_fold+'pre/train/0.csv'
# pre_couple_file_val = file_fold+'pre/val/0.csv'
pre_couple_file_test = file_fold+'pre/test/0.csv'

_, pre_train, true_train = load_pre(pre_couple_file_train)
# _, pre_val, true_val = load_pre(pre_couple_file_val)
_, pre_test, true_test = load_pre1(pre_couple_file_test)


# 计算R²和RMSE
r2_train = r2_score(true_train, pre_train)
rmse_train = np.sqrt(mean_squared_error(true_train, pre_train))

# r2_val = r2_score(true_val, pre_val)
# rmse_val = np.sqrt(mean_squared_error(true_val, pre_val))

r2_test = r2_score(true_test, pre_test)
rmse_test = np.sqrt(mean_squared_error(true_test, pre_test))

# 创建图形
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 3, 1], width_ratios=[3, 1])

ax_top_kde = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top_kde)
ax_residuals = fig.add_subplot(gs[2, 0], sharex=ax_main)
ax_right_kde = fig.add_subplot(gs[1, 1], sharey=ax_main)
# ============= 新增代码：添加点密度背景 =============

# 计算2D核密度估计
kde_train = sns.kdeplot(
    x=true_train,
    y=pre_train,
    ax=ax_main,
    fill=True,
    alpha=0.2,  # 设置透明度
    cmap="Blues",  # 使用蓝色调
    levels=20,   # 密度层级
    thresh=0.05, # 密度阈值
    zorder=1     # 确保背景在底层
)
# kde_val = sns.kdeplot(
#     x=true_val,
#     y=pre_val,
#     ax=ax_main,
#     fill=True,
#     alpha=0.2,  # 设置透明度
#     cmap="Greens",  # 使用蓝色调
#     levels=20,   # 密度层级
#     thresh=0.05, # 密度阈值
#     zorder=1     # 确保背景在底层
# )
kde_test = sns.kdeplot(
    x=true_test,
    y=pre_test,
    ax=ax_main,
    fill=True,
    alpha=0.2,  # 设置透明度
    cmap="Reds",  # 使用蓝色调
    levels=20,   # 密度层级
    thresh=0.05, # 密度阈值
    zorder=1     # 确保背景在底层
)
# ============= 新增代码结束 =============

# 添加统计信息文本（保持原有位置）
stats_text = f'R² (Training): {r2_train:.3f}\nRMSE (Training): {rmse_train:.3f}\n\nR² (Test): {r2_test:.3f}\nRMSE (Test): {rmse_test:.3f}'
ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, verticalalignment='top')

# 主散点图（带标签）- 注意添加zorder确保点在前景
sns.scatterplot(x=true_train, y=pre_train, color=palette[0], label="Training set", ax=ax_main, zorder=2)
# sns.scatterplot(x=true_val, y=pre_val, color=palette[2], label="Val set", ax=ax_main, zorder=2)
sns.scatterplot(x=true_test, y=pre_test, color=palette[3], label="Test set", ax=ax_main, zorder=2)

# 添加回归线
xlim = ax_main.get_xlim()
ylim = ax_main.get_ylim()
ax_main.plot(xlim, ylim, 'k--', alpha=0.5, lw=1.5, label='Perfect fit')  # 黑色虚线
ax_main.set_xlim(xlim)
ax_main.set_ylim(ylim)
ax_main.set_xlabel('Observed Values')
ax_main.set_ylabel('Predicted Values')
ax_main.legend(loc='lower right')

# 顶部KDE图（真实值分布）
sns.kdeplot(true_train, color=palette[0], fill=True, legend=False, ax=ax_top_kde)
# sns.kdeplot(true_val, color=palette[2], fill=True, legend=False, ax=ax_top_kde)
sns.kdeplot(true_test, color=palette[3], fill=True, legend=False, ax=ax_top_kde)

ax_top_kde.set_ylabel('Density')
ax_top_kde.set_xlabel('')
ax_top_kde.set_yticks([])
ax_top_kde.spines['top'].set_visible(False)
ax_top_kde.spines['right'].set_visible(False)
# 调整水平边缘分布图的位置
ax_top_kde.margins(y=0)  # 减少顶部和底部的空白

# 右侧KDE图（预测值分布）
sns.kdeplot(y=pre_train, color=palette[0], fill=True, legend=False,  ax=ax_right_kde)
# sns.kdeplot(y=pre_val, color=palette[2], fill=True, legend=False,  ax=ax_right_kde)
sns.kdeplot(y=pre_test, color=palette[3], fill=True, legend=False,  ax=ax_right_kde)

ax_right_kde.set_xlabel('Density')
ax_right_kde.set_ylabel('')
ax_right_kde.set_xticks([])
ax_right_kde.spines['top'].set_visible(False)
ax_right_kde.spines['right'].set_visible(False)

# 调整垂直边缘分布图的位置
ax_right_kde.margins(x=0)  # 减少左侧和右侧的空白

# 残差图
residuals_train = pre_train - true_train
# residuals_val = pre_val - true_val
residuals_test = pre_test - true_test

sns.scatterplot(x=true_train, y=residuals_train, color=palette[0], ax=ax_residuals)
# sns.scatterplot(x=true_val, y=residuals_val, color=palette[2], ax=ax_residuals)
sns.scatterplot(x=true_test, y=residuals_test, color=palette[3], ax=ax_residuals)

ax_residuals.axhline(0, color='black', linestyle='--', alpha=0.7)
ax_residuals.set_xlabel('Observed values')
ax_residuals.set_ylabel('Residuals')
ax_residuals.grid(True, linestyle=':', alpha=0.3)
# 隐藏不必要的轴
plt.setp(ax_top_kde.get_xticklabels(), visible=False)
plt.setp(ax_right_kde.get_yticklabels(), visible=False)



# 添加密度图例说明
ax_main.text(
    0.5, 0.05,
    'Darker background = Higher point density',
    transform=ax_main.transAxes,
    fontsize=9,
    ha='center',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)


ax_main.grid(False)  # 关闭主散点图的网格
ax_top_kde.grid(False)  # 关闭顶部KDE图的网格
ax_right_kde.grid(False)  # 关闭右侧KDE图的网格
ax_residuals.grid(False)  # 关闭残差图的网格

# 添加全局标题
plt.suptitle('Regression Model Performance of AF-GHGNN', fontsize=18, y=0.95)

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 保存图形为PDF文件
plt.savefig('prediction_analysis.png',dpi=1000)

# 显示图形
plt.show()

