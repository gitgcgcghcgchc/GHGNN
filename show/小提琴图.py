import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 1. 从Excel读取数据 =====
# 假设文件名为 "data.xlsx"，且数据在第一个sheet中
# Excel中两列分别为：Value, Group
df = pd.read_excel("loss_item.xlsx",sheet_name=3, header=None, names=["Value", "Group"])

# 确保Group为分类类型（便于排序）
df["Group"] = df["Group"].astype(str)
# 指定分类顺序（确保两图一致）
order = sorted(df["Group"].unique(), key=lambda x: int(x))
# print(df)
# ===== 2. 绘制小提琴图 =====
plt.figure(figsize=(6, 4))

sns.violinplot(
    data=df,
    x="Group",
    y="Value",
    order=order,
    cut=1,             # 防止“长尾”延伸
    linewidth=1.,
    color="#F1BE7C"   ##AFCBE0#BCD591

)
sns.boxplot(
    data=df,
    x="Group",
    y="Value",
    order=order,
    width=0.15,      # 控制箱线宽度
    boxprops={'zorder': 2, 'linewidth': 1.5},
    whiskerprops={'linewidth': 1.3},
    fliersize=2,
    showcaps=True,
    showfliers=False,
    color='white'
)
plt.ylim(0.5, 0.85)
# ===== 3. 添加均值折线与点 =====
# 计算每组均值
mean_df = df.groupby("Group", as_index=False)["Value"].mean()
# 将Group转为整数并按数值排序
mean_df["Group"] = mean_df["Group"].astype(int)
mean_df = mean_df.sort_values(by="Group", ascending=True)
mean_df["Group"] = mean_df["Group"].astype(str)
# print(mean_df)

# ===== 4. 手动绘制折线（对齐中心） =====
plt.plot(
    mean_df["Group"], mean_df["Value"],
    color="black",
    marker="o",
    linewidth=1.,
    markersize=3,
    label="Mean $R^2$"
)

# ===== 3. 美化 =====
plt.xlabel("Accumulated Samples", fontsize=12)
plt.ylabel(r"$R^2$", fontsize=12)
plt.title("AF-GHGNN", fontsize=14)
sns.despine()  # 去掉顶部和右侧边框
plt.tight_layout()
# 保存图形为PDF文件
plt.savefig('小提琴图AF-GHGNN.png',dpi=500, bbox_inches='tight')
plt.show()
