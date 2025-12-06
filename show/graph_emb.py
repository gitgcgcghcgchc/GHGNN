import numpy as np
from matplotlib import pyplot as plt, cm
from openpyxl.chart import marker
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import pandas as pd

def load_emb(file_name):
    """
    读取 save_emb 保存的文件，并返回：
    - protein_pairs: [(p1, p2), (p3, p4), ...]
    - h1: (N, d) numpy 数组
    - h2: (N, d) numpy 数组
    """
    proteins = []
    embeddings = []

    with open(file_name, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split(",")
            protein = parts[0]
            emb = np.array(list(map(float, parts[1].split())), dtype=np.float32)
            proteins.append(protein)
            embeddings.append(emb)

    embeddings = np.vstack(embeddings)  # (2N, d)

    # 组织成对：每两行是一个 pair
    protein_pairs = [(proteins[i], proteins[i+1]) for i in range(0, len(proteins), 2)]
    h1 = embeddings[0::2]  # (N, d)
    h2 = embeddings[1::2]  # (N, d)

    return protein_pairs, h1, h2


def load_pre(file_name):
    """
    读取 save_pre 保存的预测结果文件
    返回：
    - df: pandas.DataFrame
    - protein_pairs: [(p1, p2), ...]
    - pre: (N, 2) numpy 数组
    - true: (N, 2) numpy 数组
    - pre_couple: (N,) numpy 数组
    - true_couple: (N,) numpy 数组
    """
    df = pd.read_csv(file_name, sep="\t")
    print(df)
    # protein_pairs = list(zip(df["protein1"], df["protein2"]))
    pre = df[["pre1", "pre2"]].to_numpy()
    true = df[["true1", "true2"]].to_numpy()
    pre_couple = df["pre_couple"].to_numpy()
    true_couple = df["true_couple"].to_numpy()

    return 1,pre_couple,true_couple
def load_pre1(file_name):
    """
    读取 save_pre 保存的预测结果文件
    返回：
    - df: pandas.DataFrame
    - protein_pairs: [(p1, p2), ...]
    - pre: (N, 2) numpy 数组
    - true: (N, 2) numpy 数组
    - pre_couple: (N,) numpy 数组
    - true_couple: (N,) numpy 数组
    """
    df = pd.read_csv(file_name, sep=",")
    print(df)
    # protein_pairs = list(zip(df["protein1"], df["protein2"]))
    pre = df[["pre1", "pre2"]].to_numpy()
    true = df[["true1", "true2"]].to_numpy()
    pre_couple = df["pre_couple"].to_numpy()
    true_couple = df["true_couple"].to_numpy()

    return 1,pre_couple,true_couple
def tsne_visualize_pairs_line0(h1, h2, delta_tm, marker,
                              perplexity=50, learning_rate=300, random_state=17,
                              color_mode="clip", vmax_clip=20, quantile=95,):
    """
    t-SNE 方案二可视化：分别绘制同源蛋白，并用连线连接一对。
    线的颜色表示 ΔTm。

    参数:
    - h1, h2: (N, d) numpy 数组，每对同源蛋白的嵌入
    - delta_tm: (N,) numpy 数组，ΔTm 数值
    - color_mode: "clip" / "quantile" / "log"
    - vmax_clip: clip 模式下的上限
    - quantile: quantile 模式下的分位数
    """
    N, d = h1.shape

    # 拼成 (2N, d)，因为要对所有蛋白一起降维
    all_embeddings = abs(h1-h2)#np.concatenate([h1, h2,], axis=1)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
    embeddings_2d = tsne.fit_transform(all_embeddings)


    # ======== 颜色映射策略 ========
    if color_mode == "clip":
        vmin, vmax = delta_tm.min(), min(vmax_clip, delta_tm.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    elif color_mode == "quantile":
        vmin = np.percentile(delta_tm, 100 - quantile)  # 比如 5 分位
        vmax = np.percentile(delta_tm, quantile)  # 比如 95 分位
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    elif color_mode == "log":
        vmin = max(delta_tm.min(), 1e-6)  # 避免取 log(0)
        vmax = delta_tm.max()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    else:
        raise ValueError("color_mode must be 'clip', 'quantile', or 'log'")

    # ======== 绘制 ========
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=plt.cm.coolwarm(norm(delta_tm)), cmap='coolwarm', s=50, alpha=0.8, edgecolors='k', marker =marker)

    # 添加颜色条，表示 deltaTm 的数值大小
    cbar = plt.colorbar(scatter)
    cbar.set_label("ΔTm")

    plt.title("t-SNE visualization of GNN embeddings (ΔTm regression)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.show()

def tsne_visualize_pairs_line(h1, h2, delta_tm,
                              perplexity=50, learning_rate=300, random_state=17,
                              color_mode="clip", vmax_clip=20, quantile=95,):
    """
    t-SNE 方案二可视化：分别绘制同源蛋白，并用连线连接一对。
    线的颜色表示 ΔTm。

    参数:
    - h1, h2: (N, d) numpy 数组，每对同源蛋白的嵌入
    - delta_tm: (N,) numpy 数组，ΔTm 数值
    - color_mode: "clip" / "quantile" / "log"
    - vmax_clip: clip 模式下的上限
    - quantile: quantile 模式下的分位数
    """
    N, d = h1.shape

    # 拼成 (2N, d)，因为要对所有蛋白一起降维
    all_embeddings = h1-h2#np.vstack([h1, h2])

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    emb_h1 = embeddings_2d[:N]
    emb_h2 = embeddings_2d[N:]

    # ======== 颜色映射策略 ========
    if color_mode == "clip":
        vmin, vmax = delta_tm.min(), min(vmax_clip, delta_tm.max())

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    elif color_mode == "quantile":
        vmin = np.percentile(delta_tm, 100 - quantile)  # 比如 5 分位
        vmax = np.percentile(delta_tm, quantile)  # 比如 95 分位
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    elif color_mode == "log":
        vmin = max(delta_tm.min(), 1e-6)  # 避免取 log(0)
        vmax = delta_tm.max()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    else:
        raise ValueError("color_mode must be 'clip', 'quantile', or 'log'")

    # ======== 绘制 ========
    plt.figure(figsize=(10, 8))

    # 所有蛋白节点
    plt.scatter(embeddings_2d[:N, 0], embeddings_2d[:N, 1],
                c="blue", s=20, alpha=0.6, label="Proteins1")

    plt.scatter(embeddings_2d[N:, 0], embeddings_2d[N:, 1],
                c="red", s=20, alpha=0.6, label="Proteins2")
    # 每对蛋白之间的连线
    for i in range(N):
        plt.plot([emb_h1[i, 0], emb_h2[i, 0]],
                 [emb_h1[i, 1], emb_h2[i, 1]],
                 color=plt.cm.coolwarm(norm(delta_tm[i])),
                 alpha=0.8)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    cbar = plt.colorbar(sm)
    cbar.set_label("ΔTm")

    plt.title(f"t-SNE pairs visualization (mode={color_mode})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_fold = f'../save_result/real/HGCN_res1/PoincareBall/2025_08_13_15_15/'
    # file_fold = f'../save_result/real/GCN/Hyperboloid/2025_08_20_12_07/'
    emb_file_train = file_fold + 'emb/train/0.csv'
    emb_file_val = file_fold + 'emb/val/0.csv'
    emb_file_test = file_fold+'emb/test/0.csv'

    pre_couple_file_train = file_fold + 'pre/train/0.csv'
    pre_couple_file_val = file_fold + 'pre/val/0.csv'
    pre_couple_file_test = file_fold+'pre/test/0.csv'

    _, h1_train, h2_train = load_emb(emb_file_train)
    _, h1_val, h2_val = load_emb(emb_file_val)
    _, h1_test, h2_test = load_emb(emb_file_test)

    _,pre_train,true_train = load_pre(pre_couple_file_train)
    _,pre_val,true_val = load_pre(pre_couple_file_val)
    _,pre_test,true_test = load_pre(pre_couple_file_test)

    print(len(pre_train),len(pre_val),len(pre_test))
    delta_train = abs(pre_train - true_train)
    delta_val = abs(pre_val - true_val)
    delta_test = abs(pre_test - true_test)

    tsne_visualize_pairs_line0(h1_train, h2_train, delta_train,'o', color_mode='quantile')
    tsne_visualize_pairs_line0(h1_val,h2_val,delta_val,'^', color_mode='quantile')
    tsne_visualize_pairs_line0(h1_test,h2_test,delta_test,'D', color_mode='quantile')
