import networkx as nx
import torch
import torch.nn as nn
import torch.nn.init as init
from graphzoo.layers import HypAgg, HypAct, HypLinear



class HyperbolicGraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, manifold, c_in, dropout, use_att, local_agg, use_bias=True,):

        super(HyperbolicGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.agg = HypAgg(manifold, c_in, output_dim, dropout, use_att, local_agg)
        self.linear = HypLinear(manifold, input_dim, output_dim, c_in, dropout, use_bias)


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
        support = self.linear.forward(input_feature)
        output =self.agg(support, adjacency)
        #这一步是否有必要？
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


def calculate_hyperbolic_distance(adjacency_matrix, ):
    """
    计算每个节点到图中中心的距离。
    假设中心节点是度最大的节点。
    """
    # 使用 NetworkX 来计算图的中心
    G = nx.from_scipy_sparse_matrix(adjacency_matrix)
    degree_centrality = nx.degree_centrality(G)

    # 选择度最大的节点作为中心节点
    central_node = max(degree_centrality, key=degree_centrality.get)

    # 使用 NetworkX 的最短路径算法来计算节点到中心的最短路径
    shortest_paths = nx.single_source_shortest_path_length(G, central_node)

    # 将结果转换为 PyTorch Tensor
    distances = torch.zeros(adjacency_matrix.shape[0])
    for node, distance in shortest_paths.items():
        distances[node] = distance

    return distances

def normalize_hyperbolic_distances(distances):
    """
    根据距离对节点进行归一化，使其适应双曲空间的度量。
    """
    max_distance = torch.max(distances)
    normalized_distances = distances / max_distance
    return normalized_distances

def top_rank(attention_score, keep_ratio):
    # 获取节点总数
    num_nodes = len(attention_score)

    # 计算需要保留的节点数量
    keep_node_num = int(keep_ratio * num_nodes)

    # 对注意力分数进行降序排序，获取排序后的索引
    _, sorted_index = attention_score.sort(descending=True)

    # 创建一个布尔掩码，初始化为 False
    mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 将排序后前 keep_node_num 个节点对应的掩码值设置为 True
    mask[sorted_index[:keep_node_num]] = True

    return mask


def global_max_pool_hyperbolic(x, distances):
    """
    双曲空间中的最大池化。根据节点到中心的几何距离选择最相关的节点特征。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (num_nodes, num_features)。
        distances (torch.Tensor): 节点到中心节点的距离，形状为 (num_nodes,)。

    返回:
        torch.Tensor: 聚合后的特征向量，形状为 (1, num_features)。
    """
    # 根据节点到中心的距离加权节点特征
    # 在双曲空间中，距离越近的节点可能更重要，因此我们可以使用距离的负值来加权特征
    weight = torch.exp(-distances)  # 近距离节点权重大，远距离权小

    # 对节点特征进行加权
    weighted_x = x * weight.view(-1, 1)

    # 最大池化，选择加权后的最大特征值
    result = weighted_x.max(dim=0, keepdim=True)[0]

    return result


def global_avg_pool_hyperbolic(x, distances):
    """
    双曲空间中的平均池化。根据节点到中心的几何距离加权平均节点特征。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (num_nodes, num_features)。
        distances (torch.Tensor): 节点到中心节点的距离，形状为 (num_nodes,)。

    返回:
        torch.Tensor: 聚合后的特征向量，形状为 (1, num_features)。
    """
    # 根据节点到中心的距离加权节点特征
    weight = torch.exp(-distances)  # 近距离节点权重大，远距离权小

    # 对节点特征进行加权
    weighted_x = x * weight.view(-1, 1)

    # 计算加权平均池化
    result = weighted_x.sum(dim=0, keepdim=True) / weight.sum()

    return result



class HyperbolicSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(HyperbolicSelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = HyperbolicGraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature):
        # 计算注意力分数
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        # 调整注意力分数，考虑到距离中心的度量
        distances = calculate_hyperbolic_distance(input_feature)
        adjusted_attn_score = self.adjust_for_hyperbolic(attn_score, distances)
        attn_score = self.activation(adjusted_attn_score)

        mask = top_rank(attn_score, self.keep_ratio)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)

        return hidden

