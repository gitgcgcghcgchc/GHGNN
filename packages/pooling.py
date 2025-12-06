
import torch
import torch.nn as nn
import torch.nn.init as init

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



def top_rank(attention_score, keep_ratio):
    """
    attention_score: (B, N) Tensor
    keep_ratio: float, ratio of nodes to keep
    return: (B, N) BoolTensor mask
    """
    attention_score = attention_score.unsqueeze(0)
    B, N = attention_score.shape
    keep_node_num = max(int(keep_ratio * N), 1)  # 每图至少保留1个节点

    # 初始化 mask 全为 False
    mask = torch.zeros_like(attention_score, dtype=torch.bool)

    # 对每个图分别处理
    for i in range(B):
        _, sorted_index = attention_score[i].sort(descending=True)
        k = min(keep_node_num, sorted_index.size(0))  # 防止越界
        mask[i, sorted_index[:k]] = True

    return mask

def global_max_pool(x):
    """
    对整个图的节点特征进行全局最大池化操作。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (num_nodes, num_features)。

    返回:
        torch.Tensor: 聚合后的特征向量，形状为 (1, num_features)。
    """
    # 对整个图的节点特征进行最大值聚合
    result = x.max(dim=1, keepdim=True)[0]
    return result

def global_avg_pool(x):
    """
    对整个图的节点特征进行全局平均池化操作。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (num_nodes, num_features)。

    返回:
        torch.Tensor: 聚合后的特征向量，形状为 (1, num_features)。
    """
    # 对整个图的节点特征进行平均值聚合
    result = x.mean(dim=1, keepdim=True)
    return result



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature):
        # 计算注意力分数
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)

        mask = top_rank(attn_score, self.keep_ratio)
        hidden = (input_feature*mask.unsqueeze(-1).float())*(attn_score.unsqueeze(-1))

        return hidden

