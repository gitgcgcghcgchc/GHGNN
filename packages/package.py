import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


def top_rank(attention_score, graph_indicator, keep_ratio):

    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score = attention_score[graph_indicator == graph_id]
        graph_node_num = len(graph_attn_score)
        graph_mask = attention_score.new_zeros((graph_node_num,), dtype=torch.bool)
        keep_graph_node_num = int(keep_ratio * graph_node_num)
        _, sorted_index = graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[:keep_graph_node_num]] = True
        mask = torch.cat((mask, graph_mask))

    return mask


def normalization(adjacency):
    adjacency += adjacency.T
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse_coo_tensor(indices, values, L.shape)
    return tensor_adjacency


def filter_adjacency(adjacency, mask):

    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)


def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1  # 获取图的数量
    result = torch.zeros(num, x.size(1), device=x.device)  # 创建一个结果张量

    # 对每个图进行最大值聚合
    for i in range(num):
        # 获取属于当前图的节点索引
        mask = (graph_indicator == i)

        # 如果当前图中有节点，则进行最大值聚合
        if mask.sum() > 0:
            result[i] = x[mask].max(dim=0)[0]

    return result


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1  # 获取图的数量
    result = torch.zeros(num, x.size(1), device=x.device)  # 创建一个结果张量
    counts = torch.zeros(num, device=x.device)  # 用于计算每个图中节点的数量

    # 对每个图进行平均值聚合
    for i in range(num):
        # 获取属于当前图的节点索引
        mask = (graph_indicator == i)

        # 如果当前图中有节点，则进行平均值聚合
        if mask.sum() > 0:
            result[i] = x[mask].mean(dim=0)
            counts[i] = mask.sum()

    # 对于每个图，计算节点的平均值
    return result



def mean_pool(tensor, graph_indicator):


    # 获取唯一的图ID
    unique_graph_ids = torch.unique(graph_indicator)

    pooled_results = []

    # 对每个图进行池化
    for graph_id in unique_graph_ids:
        # 获取属于该图的所有节点的行索引
        node_indices = (graph_indicator == graph_id).nonzero().squeeze()

        # 对属于该图的节点行进行平均池化
        graph_nodes = tensor[node_indices]
        pooled_result = graph_nodes.mean(dim=0)  # 按列取平均

        # 将池化结果加入结果列表
        pooled_results.append(pooled_result)

    # 将所有图的池化结果堆叠成一个张量
    pooled_results = torch.stack(pooled_results)

    return pooled_results

def symmetric_normalize_adj_matrix(adj_matrix):
    """
    对邻接矩阵进行对称归一化。
    输入可以是单个邻接矩阵（形状为 [N, N]）或多个邻接矩阵（形状为 [K, N, N]）。
    """
    # 检查输入形状
    if len(adj_matrix.shape) == 2:
        # 单个邻接矩阵
        assert adj_matrix.shape[0] == adj_matrix.shape[1], "邻接矩阵必须是方阵"
        return _normalize_single(adj_matrix)
    elif len(adj_matrix.shape) == 3:
        # 多个邻接矩阵
        assert adj_matrix.shape[1] == adj_matrix.shape[2], "每个邻接矩阵必须是方阵"
        return _normalize_multiple(adj_matrix)
    else:
        raise ValueError("输入的邻接矩阵形状不正确，必须是 [N, N] 或 [K, N, N]")

def _normalize_single(adj_matrix):
    # 计算每个节点的度
    row_sum = adj_matrix.sum(dim=1)
    D_inv_sqrt = row_sum.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0

    # 构造对角矩阵 D^{-1/2}
    D_inv_sqrt_matrix = torch.diag(D_inv_sqrt)

    # 对称归一化
    normalized_adj_matrix = D_inv_sqrt_matrix @ adj_matrix @ D_inv_sqrt_matrix
    return normalized_adj_matrix

def _normalize_multiple(adj_matrix):
    # 获取每个邻接矩阵的大小
    K, N, _ = adj_matrix.shape

    # 计算每个节点的度（对每个邻接矩阵分别计算）
    row_sum = adj_matrix.sum(dim=2)  # 形状为 [K, N]
    D_inv_sqrt = row_sum.pow(-0.5)  # 形状为 [K, N]
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0

    # 构造对角矩阵 D^{-1/2}（对每个邻接矩阵分别构造）
    D_inv_sqrt_matrix = torch.diag_embed(D_inv_sqrt)  # 形状为 [K, N, N]

    # 对称归一化（对每个邻接矩阵分别归一化）
    normalized_adj_matrix = D_inv_sqrt_matrix @ adj_matrix @ D_inv_sqrt_matrix
    return normalized_adj_matrix


class Dataset(object):
    def __init__(self,
                 data_root="data",
                 data_name='test',
                 edge_file='test.txt',
                 node_labels_file='test_node_labels.txt',
                 node_attributes_file=None,
                 indicator_file='test_graph_indicator.txt',
                 graph_label_file='test_graph_labels.txt',
                 seed = 65):

        self.data_name = data_name
        self.edge_file = edge_file
        self.node_labels_file = node_labels_file
        self.node_attributes_file = node_attributes_file
        self.indicator_file = indicator_file
        self.graph_label_file = graph_label_file
        self.data_root = data_root
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        self.node_attributes = self.load_node_attributes()
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.node_attributes = None
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        self.train_index, self.test_index = train_test_split(np.arange(len(graph_labels)), test_size=0.1,
                                                             random_state=seed)
        self.train_label = graph_labels[self.train_index]
        self.test_label = graph_labels[self.test_index]
        self.train_val_indices = np.array_split(self.train_index, 10)

    def get_train_val_splits(self):
        for i in range(10):
            val_index = self.train_val_indices[i]
            train_index = np.concatenate([self.train_val_indices[j] for j in range(10) if j != i])
            train_label = self.graph_labels[train_index]
            val_label = self.graph_labels[val_index]
            yield train_index, train_label, val_index, val_label, self.test_index, self.test_label

    def __getitem__(self, index):
        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        node_attributes = self.node_attributes[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels,node_attributes, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)

    def read_data(self):
        data_dir = os.path.join(self.data_root, self.data_name, )
        #print("Loading edge_file...")
        adjacency_list = np.genfromtxt(os.path.join(data_dir, self.edge_file),
                                       dtype=np.int64, delimiter=',') - 1
        #print("Loading node_labels_file...")
        node_labels = np.genfromtxt(os.path.join(data_dir, self.node_labels_file),
                                    dtype=np.int64) - 1
        #print("Loading graph_indicator_file...")
        graph_indicator = np.genfromtxt(os.path.join(data_dir, self.indicator_file),
                                        dtype=np.int64) - 1
        #print("Loading graph_labels_file...")
        graph_labels = np.genfromtxt(os.path.join(data_dir, self.graph_label_file),
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)),
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        #print("The number of nodes is: ", num_nodes)
        return sparse_adjacency, node_labels, graph_indicator, graph_labels

    def load_node_attributes(self):
        data_dir = os.path.join(self.data_root, self.data_name)
        if self.node_attributes_file is not None:
            #print("Loading node_attributes_file...")
            self.node_attributes = np.genfromtxt(os.path.join(data_dir, self.node_attributes_file),
                                                 dtype=np.float32, delimiter=',')

    def process_node_features(self, node_features):
        if self.node_attributes is not None:
            if self.node_attributes.ndim == 1:
                node_attributes_2d = self.node_attributes[:, np.newaxis]
            else:
                node_attributes_2d = self.node_attributes
            node_features = torch.from_numpy(np.concatenate((node_features, node_attributes_2d), axis=1))
        return node_features
