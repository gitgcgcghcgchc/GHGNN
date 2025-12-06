import os
import re

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, TruncatedSVD


def get_species(species_path):
    speciess = {}
    k={}
    k["ESCHERICHIA COLI"]=0
    k["HOMO SAPIENS"]=0
    k["SACCHAROMYCES CEREVISIAE"]=0
    k["THERMUS THERMOPHILUS"] = 0
    k["NEOVISON VISON"] = 0
    m=0
    for root, dirs, species_files in os.walk(species_path):
        for species_file in species_files:
            # 将文件名（不包含扩展名）作为键，标签作为值存储在字典中
            network_name = os.path.splitext(species_file)[0]
            # print(network_name)

            # print(lable_file)
            with open(species_path + "/" + species_file, 'r') as file:
                lines = file.readlines()[5:12]
                a=lines[0]+lines[1]+lines[2]+lines[3]+lines[4]+lines[5]
                if "ESCHERICHIA COLI" in a:
                    species = 1
                    k["ESCHERICHIA COLI"]+=1

                elif "HOMO SAPIENS" in a:
                    species = 2
                    k["HOMO SAPIENS"] += 1

                elif "SACCHAROMYCES CEREVISIAE" in a:
                    species = 3
                    k["SACCHAROMYCES CEREVISIAE"] += 1

                elif "THERMUS THERMOPHILUS" in a:
                    species = 4
                    k["THERMUS THERMOPHILUS"] += 1
                elif "NEOVISON VISON" in a:
                    species = 5
                    k["NEOVISON VISON"] += 1
                else:
                    species = 0
                    m=m+1
                    #print(network_name)
            # print(label)
            if species != 'N.vison':
                speciess[network_name[3:9]] = species
    #print(k)
    return speciess

def get_network(path):
    name=[]
    seq={}
    networks = {}
    networks_adj = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            # 将文件名（不包含扩展名）作为键，标签作为值存储在字典中
            network_name = os.path.splitext(file)[0][3:9]
            name.append(network_name)
            seq_p={}

            result = []  # 用于存储最终结果的列表
            with open(path + "/" + file, 'r', encoding='utf-8') as file:
                for line in file:
                    # 去掉行首行尾的空白符，并以空格分隔每一行的元素
                    elements = line.strip().split()
                    elements0 = []
                    for string in elements:

                        # 使用正则表达式提取数字

                        numbers = re.findall(r"[-+]?\d*\.?\d+", string)
                        seq_i = string[-3:]
                        # 将提取的数字转换为 float 或 int
                        for num in numbers:
                            if '.' in num:
                                numbers = float(num)
                            else:
                                numbers = int(num)
                                seq_p[num] = seq_i
                        elements0.append(numbers)
                    # 将分隔后的元素列表添加到结果列表中
                    result.append(elements0)
            # 创建无向图
            G = nx.Graph()

            # 添加边和权值
            for u, v, weight in result:
                G.add_edge(u, v, weight=weight)

            # 获取节点列表（按顺序排列）
            nodes = sorted(G.nodes())

            # 创建邻接矩阵
            num_nodes = len(nodes)
            adj_matrix = np.zeros((num_nodes, num_nodes))

            # 填充邻接矩阵
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                adj_matrix[u-1][v-1] = weight
                adj_matrix[v-1][u-1] = weight  # 无向图，对称
            networks[network_name] = result
            networks_adj[network_name] = adj_matrix
            seq[network_name] = seq_p
    return name, networks,networks_adj, seq

def make_seq(seq):

    dict={
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    for p_name,p_dict in seq.items():
        p_list = [dict[p_dict[key]] for key in sorted(p_dict.keys(), key=int)]
        p_string = ''.join(p_list)
        seq[p_name]= p_string
    return seq


def get_matrix(folder_path):
    # 初始化字典
    feature_matrices = {}

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        exit()

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):  # 确保只处理.npy文件
            # 构建文件路径
            file_path = os.path.join(folder_path, file_name)

            # 加载矩阵
            matrix = np.load(file_path)

            # 文件名作为键（去掉.npy扩展名）
            protein_name = os.path.splitext(file_name)[0]
            # 添加到字典
            feature_matrices[protein_name] = coo_matrix(matrix, dtype=np.float32)
    return feature_matrices

def save_adj(adj_matrices,folder_path):
    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)

    # 保存每个邻接矩阵到文件
    for network_name, adj_matrix in adj_matrices.items():
        file_path = os.path.join(folder_path, f'{network_name}.npy')
        np.save(file_path, adj_matrix)
        print(f'已保存 {network_name} 的邻接矩阵到 {file_path}')

def save_node_features(protein_sequences, folder_path):

    use_aaindex=0

    # 定义氨基酸类型
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # 创建氨基酸到one-hot编码的映射
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    num_amino_acids = len(amino_acids)

    # 初始化字典来存储one-hot编码矩阵
    one_hot_matrices = {}

    # 创建文件夹

    os.makedirs(folder_path, exist_ok=True)

    # 遍历每个蛋白质序列
    for protein_name, sequence in protein_sequences.items():
        # 初始化one-hot编码矩阵
        one_hot_matrix = np.zeros((len(sequence), num_amino_acids))

        # 填充矩阵
        for i, aa in enumerate(sequence):
            if aa in aa_to_index:
                one_hot_matrix[i, aa_to_index[aa]] = 1
            else:
                # 如果氨基酸不在定义的列表中，可以处理为全零或报错
                print(f"警告：未知氨基酸 '{aa}' 在蛋白质 '{protein_name}' 中，跳过。")

        # 保存到字典
        one_hot_matrices[protein_name] = one_hot_matrix

        # 保存到文件
        file_path = os.path.join(folder_path, f'{protein_name}.npy')
        np.save(file_path, one_hot_matrix)
        print(f'已保存 {protein_name} 的one-hot编码矩阵到 {file_path}')


def save_species(species_dict,file_path):
    # 指定文件路径

    # 保存字典到文件
    with open(file_path, 'w') as f:
        for protein_name, species_id in species_dict.items():
            f.write(f"{protein_name},{species_id}\n")

    print(f"已将字典保存到 {file_path}")

def get_specieses(species_path):
    # 初始化字典
    loaded_species_dict = {}

    # 读取文件并构建字典
    with open(species_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保行不为空
                protein_name, species_id = line.split(',')
                loaded_species_dict[protein_name] = int(species_id)
    return loaded_species_dict

def get_AAindex(filepath):
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 从第二行开始处理
    for line in lines[1:]:
        parts = line.strip().split()  # 默认以空格或多个空白字符分隔
        if not parts:
            continue  # 跳过空行
        key = parts[0]
        values = list(map(float, parts[1:]))
        result[key] = values
    return result


def pca_reduce_aaindex(filepath, n_components=10):
    aa_dict = get_AAindex(filepath)
    aa_keys = list(aa_dict.keys())
    feature_matrix = np.array([aa_dict[aa] for aa in aa_keys])

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(feature_matrix)


    reduced_dict = {aa_keys[i]: reduced[i].tolist() for i in range(len(aa_keys))}

    # print("累计解释的方差比例：", np.sum(pca.explained_variance_ratio_))
    return reduced_dict

def get_aaindex_feature(sequence, aaindex_pca):
    k=len(aaindex_pca['A'])
    one_hot_matrix = np.zeros((len(sequence), k))

    # 填充矩阵
    for i, aa in enumerate(sequence):
        one_hot_matrix[i][:]=aaindex_pca[aa]
    # print(one_hot_matrix)

    return one_hot_matrix

def floyd_warshall(adj_matrixs):
    """
    输入：
        adj_matrix: 一个 n x n 的加权邻接矩阵（2D list 或 numpy array），
                    如果 i 和 j 之间没有边，应该用 np.inf 表示。
    输出：
        dist: 最短路径矩阵，dist[i][j] 是从节点 i 到 j 的最短距离。
    """
    # 将输入矩阵转换为 numpy array 以便处理
    dists = {}
    for protein_name, adj_matrix in adj_matrixs.items():
        dists[protein_name] = shortest_path(csgraph=adj_matrix,
                                    directed=True,
                                    unweighted=False,
                                    return_predecessors=False)

    return dists