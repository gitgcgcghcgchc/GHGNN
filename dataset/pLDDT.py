import pandas as pd
from matplotlib import pyplot as plt

# # 读取species.txt文件中的蛋白质名称
# with open('StructΔTm801/species.txt', 'r') as file:
#     species_lines = file.readlines()
#
# # 提取蛋白质名称，假设每行的第一个值是蛋白质名称
# species = [line.strip().split(',')[0] for line in species_lines]

# 读取real_couple.xlsx文件
real_couple_df = pd.read_excel('StructΔTm801/real_couple.xlsx',skiprows=1)

# 提取前两列中的所有蛋白质名称
species = pd.concat([real_couple_df['protein1'], real_couple_df['protein2']]).unique()


print(len(species))
# 读取results.csv文件
results_df = pd.read_csv('plddt_screening_results.csv')
# print(results_df)
# 筛选出results中在species中的蛋白质名称对应的行
filtered_results = results_df[results_df['Protein_ID'].isin(species)]
print(filtered_results)
# 将筛选结果保存到新的CSV文件中
filtered_results.to_csv('filtered_results.csv', index=False)

print("筛选完成，结果已保存到filtered_results.csv文件中。")
# 定义pLDDT的分组区间
bins = [0, 70, 80, 90, 100]  # 你可以根据需要调整这些区间
labels = ['0-69', '70-79', '80-89', '90-100']  # 对应的标签

# 使用cut函数将pLDDT值分到指定的区间中
filtered_results['pLDDT_group'] = pd.cut(filtered_results['pLDDT'], bins=bins, labels=labels, right=False)

# 统计每个区间的频率
pLDDT_group_counts = filtered_results['pLDDT_group'].value_counts().sort_index()


# 绘制柱状图
plt.figure(figsize=(10, 6))
ax=pLDDT_group_counts.plot(kind='bar', color='skyblue')
plt.title('pLDDT Value Distribution')
plt.xlabel('pLDDT Value')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
# 在每个柱子上显示蛋白质的数量
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()
