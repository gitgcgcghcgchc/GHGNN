import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.lines import Line2D
from openpyxl.chart import marker
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from delta_tm.dataset.packages import get_network, make_seq
import warnings
warnings.filterwarnings('ignore')

def load_node_emb(folder_path):
    """
    读取指定文件夹中的所有.npy文件，并将它们存储到一个字典中。
    :param folder_path: 包含.npy文件的文件夹路径
    :return: 一个字典，键是文件名，值是对应的矩阵
    """
    npy_dict = {}
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否是.npy文件
        if file_name.endswith('.npy'):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            # 读取.npy文件
            matrix = np.load(file_path)
            # 将文件名（不带扩展名）作为键，矩阵作为值存储到字典中
            npy_dict[file_name[:-4]] = matrix
    return npy_dict

def plot_amino_acid_composition(protein_sequences):
    """
    绘制多个蛋白质的氨基酸组成条形图。

    参数:
    protein_sequences (dict): 一个字典，键是蛋白质名称，值是蛋白质序列。

    返回:
    None
    """
    # 定义氨基酸的顺序
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 创建一个字典来存储每种蛋白质的氨基酸计数
    amino_acid_counts = {protein: Counter(sequence) for protein, sequence in protein_sequences.items()}

    # 创建一个条形图
    fig, ax = plt.subplots()

    # 为每种蛋白质绘制条形图
    bar_width = 0.35
    for i, (protein, counts) in enumerate(amino_acid_counts.items()):
        # 按照氨基酸的顺序排列计数
        counts_ordered = [counts.get(amino_acid, 0) for amino_acid in amino_acids]
        ax.bar([x + i * bar_width for x in range(len(amino_acids))], counts_ordered, width=bar_width, label=protein)

    # 设置x轴标签
    ax.set_xticks(range(len(amino_acids)))
    ax.set_xticklabels(amino_acids)

    # 添加标题和标签
    ax.set_title('Amino Acid Composition of Proteins')
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Count')
    ax.legend()

    # 显示图形
    plt.show()

def plot_top_n_residue_counts(df_sorted, n=100):
    """
    统计重要性排名前n的残基中各种类型残基的数量，并绘制条形图。

    参数:
    df_sorted (pd.DataFrame): 已经按照重要性排序的DataFrame，包含列 'Amino_Acid' 和 'Importance'。
    n (int): 要统计的残基数量，默认为100。
    """
    # 定义氨基酸的顺序
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 获取重要性排名前n的残基
    top_n = df_sorted.head(n)

    # 统计每种残基的数量
    residue_counts = top_n['Amino_Acid'].value_counts().reset_index()
    residue_counts.columns = ['Amino_Acid', 'Count']

    # 按照定义的氨基酸顺序排序
    residue_counts['Order'] = residue_counts['Amino_Acid'].apply(lambda x: amino_acids.index(x) if x in amino_acids else len(amino_acids))
    residue_counts.sort_values(by='Order', inplace=True)
    residue_counts.drop(columns=['Order'], inplace=True)

    # 打印统计结果
    print(residue_counts)

    # 条形图可视化
    plt.figure(figsize=(12, 8))
    plt.bar(residue_counts['Amino_Acid'], residue_counts['Count'], color='skyblue')
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')
    plt.title(f'Top {n} Amino Acids Count')
    plt.xticks(rotation=90)  # 旋转x轴标签，避免重叠
    plt.tight_layout()  # 调整布局
    plt.show()

def tsne_visualize_embeddings(key, node_embeddings, node_types_str, node_importance, perplexity=50, learning_rate=1000, random_state=42):
    """
    使用TSNE对节点嵌入矩阵进行降维并可视化。
    :param node_embeddings: 节点嵌入矩阵 (numpy数组)
    :param node_types_str: 每个节点的类型 (字符串，每个字符代表一个节点的类型)
    :param node_importance: 每个节点的重要性 (浮点数数组)
    :param perplexity: TSNE的困惑度参数 (默认30)
    :param learning_rate: TSNE的学习率参数 (默认200)
    :param random_state: 随机种子 (默认42)
    :return: None
    """
    # 将字符串转换为列表
    node_types = list(node_types_str)

    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state, init='random')
    node_embeddings_2d = tsne.fit_transform(node_embeddings)

    # 创建一个DataFrame来存储降维后的数据和节点类型、重要性
    data = pd.DataFrame(node_embeddings_2d, columns=['x', 'y'])
    data['type'] = node_types
    data['importance'] = node_importance

    # 裁剪重要性值以避免异常值的影响
    q_low = np.percentile(node_importance, 5)  # 5% 分位数
    q_high = np.percentile(node_importance, 95)  # 95% 分位数
    node_importance_clipped = np.clip(node_importance, q_low, q_high)

    # 定义节点形状和颜色
    unique_types = np.unique(node_types)
    markers =['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h',
           'H', '+', 'x', 'd', '|', '_', '.', '1', '2', '3']
    type_shapes = {node_type: shape for node_type, shape in zip(unique_types, markers)}

    # 归一化重要性值
    scaler = MinMaxScaler()
    normalized_importance = scaler.fit_transform(node_importance_clipped.reshape(-1, 1)).flatten()
    importance_colors = plt.cm.viridis(normalized_importance)  # 根据归一化后的值映射颜色

    # 绘制TSNE可视化图
    plt.figure(figsize=(10, 8))
    for node_type, shape in type_shapes.items():
        subset = data[data['type'] == node_type]
        plt.scatter(subset['x'], subset['y'], c=importance_colors[subset.index],
                    facecolors='none',
                    marker=shape, label=node_type, edgecolors='w', s=100)

    # 创建自定义图例句柄
    legend_handles = []
    for i, shape in enumerate(markers):
        legend_handles.append(Line2D([0], [0], marker=shape, color='w', label=unique_types[i],
                                     markerfacecolor='lightgray', markeredgecolor='black', markersize=10))


    # 添加图例
    plt.legend(title='Node Type', loc='upper right', ncol=2, handles=legend_handles,)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(normalized_importance), vmax=max(normalized_importance))),
                 label='Node Importance')
    plt.title(f't-SNE Visualization of Node Embeddings for {key}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # 保存图形为PDF文件
    plt.savefig(f'TSNE Visualization of Node Embeddings for {key}.png', dpi=500)
    plt.show()

if __name__ == "__main__":
    # file_fold = f'../save_result/real/HGCN_res1/PoincareBall/2025_09_02_16_06/'
    file_fold = f'../save_result/StructΔTm801/GHG/PoincareBall/2025_10_09_15_47/'


    node_emb_file = file_fold+'nod_emb/test/0'
    pool_score_file = file_fold+'pool_score/test/0'

    name, networks, networks_adj, seq = get_network(f"../dataset/StructΔTm801/wl")
    seq = make_seq(seq)
    seq = {key: value for key, value in seq.items() if key in ('P04391','P05150')}
    # seq = {key: value for key, value in seq.items() if key in ('P56690', 'P41252')}


    # plot_amino_acid_composition(seq)

    node_embs = load_node_emb(node_emb_file)

    pool_scores = load_node_emb(pool_score_file)

    # for key in ['P56690', 'P41252']:
    for key in ['P04391','P05150']:

        # 创建一个DataFrame
        df = pd.DataFrame({
            'Amino_Acid': list(seq[key]),
            'Importance': pool_scores[key]
        })
        df['Original_Index'] = df.index
        # print(seq[key])
        # print(df)
        # 按照重要性从大到小排序
        df_sorted = df.sort_values(by='Importance', ascending=False)
        # print(df_sorted)
        # 重置索引
        df_sorted.reset_index(drop=True, inplace=True)
        # plot_top_n_residue_counts(df_sorted, n=100)
        # 打印结果
        print(df_sorted[:20])

        tsne_visualize_embeddings(key, node_embs[key], node_types_str=seq[key], node_importance=pool_scores[key])
