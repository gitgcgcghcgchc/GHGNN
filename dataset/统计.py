import pandas as pd

from graph_data_reader_couple import get_data

df,splits=get_data(root_file='')
# print(df)
df1=df[['protein1','TM1','species1','feature1','seq1']].rename(columns={
    'protein1': 'protein',
    'TM1': 'TM',
    'species1': 'species',
    'feature1': 'feature',
    'seq1': 'seq',
})
df2=df[['protein2','TM2','species2','feature2','seq2']].rename(columns={
    'protein2': 'protein',
    'TM2': 'TM',
    'species2': 'species',
    'feature2': 'feature',
    'seq2': 'seq',
})

# 合并两个DataFrame
merged_df = pd.concat([df1, df2])

# 根据'protein'列去除重复项，保留第一个出现的行
final_df = merged_df.drop_duplicates(subset=['protein'], keep='first').reset_index(drop=True)

# 统计species的种类及出现次数
species_counts = final_df['species'].value_counts()
# 计算每个物种的比例
species_proportion = species_counts / species_counts.sum()
print("\n物种的种类及比例：")
print(species_proportion)

# 计算TM列的范围
tm_range = df['delta_TM'].max() - df['delta_TM'].min()
# 计算TM列的平均值
tm_mean = df['delta_TM'].mean()
# 计算TM列的方差
tm_variance = df['delta_TM'].var()

# 输出结果
print("TM列的统计结果：")
print(f"[{df['delta_TM'].min()}~{df['delta_TM'].max()}]")
print(f"范围 (Range): {tm_range}")
print(f"平均值 (Mean): {tm_mean}")
print(f"方差 (Variance): {tm_variance**0.5}")


final_df['len'] = final_df['seq'].str.len()
print(f"{final_df['len'].min()}~{final_df['len'].max()}")