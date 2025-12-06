import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 1. 读取 Excel 数据 =====
# 假设文件名为 'data.xlsx'，两列分别为 'R2' 和 'Group'
df1 = pd.read_excel('dim.xlsx', sheet_name=1, header=None, names=['R2', 'Group'])
df2 = pd.read_excel('dim.xlsx', sheet_name=2, header=None, names=['R2', 'Group'])
df3 = pd.read_excel('dim.xlsx', sheet_name=3, header=None, names=['R2', 'Group'])
df1["Group"] = df1["Group"].astype(str)
df2["Group"] = df2["Group"].astype(str)
df3["Group"] = df3["Group"].astype(str)

# ===== 2. 按组计算均值与标准差 =====
stats1 = df1.groupby('Group')['R2'].agg(['mean', 'std']).reset_index()
stats2 = df2.groupby('Group')['R2'].agg(['mean', 'std']).reset_index()
stats3 = df3.groupby('Group')['R2'].agg(['mean', 'std']).reset_index()

stats1["Group"] = stats1["Group"].astype(int)
stats1 = stats1.sort_values(by="Group", ascending=True)
stats1["Group"] = stats1["Group"].astype(str)

stats2["Group"] = stats2["Group"].astype(int)
stats2 = stats2.sort_values(by="Group", ascending=True)
stats2["Group"] = stats2["Group"].astype(str)

stats3["Group"] = stats3["Group"].astype(int)
stats3 = stats3.sort_values(by="Group", ascending=True)
stats3["Group"] = stats3["Group"].astype(str)

# ===== 3. 绘图 =====
plt.figure(figsize=(6,4), dpi=120)


# 绘制均值折线
plt.plot(stats1['Group'], stats1['mean'], color='#3E75AE', linewidth=2.2, marker='o', label='Mean R² of GCN')
plt.plot(stats2['Group'], stats2['mean'], color='#549C52', linewidth=2.2, marker='o', label='Mean R² of HGNN')
plt.plot(stats3['Group'], stats3['mean'], color='#E7853E', linewidth=2.2, marker='o', label='Mean R² of AF-GHGNN')
# 绘制标准差阴影区域
plt.fill_between(stats1['Group'], stats1['mean'] - stats1['std'], stats1['mean'] + stats1['std'], color='#AFCBE0', alpha=0.2, label='±1 Std. Dev. of GCN',)
plt.fill_between(stats2['Group'], stats2['mean'] - stats2['std'], stats2['mean'] + stats2['std'], color='#BCD591', alpha=0.2, label='±1 Std. Dev. of HGNN',)
plt.fill_between(stats3['Group'], stats3['mean'] - stats3['std'], stats3['mean'] + stats3['std'], color='#F1BE7C', alpha=0.2, label='±1 Std. Dev. of AF-GHGNN',)
#colors = ['#AFCBE0', '#BCD591', '#F1BE7C']
# ===== 4. 美化图像 =====
plt.xlabel('Dimension')
plt.ylabel('R²')
# plt.grid(alpha=0.3)
plt.legend(frameon=True, fontsize=8)
plt.tight_layout()
plt.savefig("R2_comparison.png", dpi=500, bbox_inches='tight')
plt.show()
