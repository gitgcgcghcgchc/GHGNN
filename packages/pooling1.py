import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.init as init
from graphzoo import manifolds
from graphzoo.layers import DenseAtt


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

class HypGraphConvolution(nn.Module):
    """
    Hyperbolic aggregation layer
    """

    def __init__(self, manifold, c, dropout, input_dim, output_dim, use_bias=True):
        super(HypGraphConvolution, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.c = c
        self.use_bias = use_bias
        self.dropout = dropout
        if self.manifold.name == 'Hyperboloid':
            bias = torch.Tensor(output_dim)
            o = torch.zeros_like(bias)
            self.bias = nn.Parameter(torch.cat([o,bias], dim=0))
        else:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        # def check_nan(tensor, name):
        #     if torch.isnan(tensor).any():
        #         print(f"⚠️ 检测到 NaN：{name}")
        # check_nan(x, name='x')
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # check_nan(self.weight, 'weight')
        # check_nan(x_tangent, 'x_tangent')
        support = torch.mm(x_tangent, self.weight)
        # check_nan(support, 'support')
        if self.use_bias is True:
            bias = self.manifold.proj_tan0(self.bias.view(1,-1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(support, hyp_bias, c=self.c)
            support = self.manifold.proj(res, self.c)

        output = torch.sparse.mm(adj, support)
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)

        return output


    def extra_repr(self):
        return 'c={}'.format(self.c)

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

def global_max_pool(x):
    """
    对整个图的节点特征进行全局最大池化操作。

    参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (num_nodes, num_features)。

    返回:
        torch.Tensor: 聚合后的特征向量，形状为 (1, num_features)。
    """
    # 对整个图的节点特征进行最大值聚合
    result = x.max(dim=0, keepdim=True)[0]
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
    result = x.mean(dim=0, keepdim=True)
    return result



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, args, c, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        # self.attn_gcn = GraphConvolution(input_dim, 1)
        self.attn_gcn = HypGraphConvolution(args.manifold, c, args.dropout, input_dim, 1, use_bias=True)

        self.manifold = getattr(manifolds, args.manifold)()
        self.c = c

        # ✅ 添加可学习融合参数（初始化为平均）
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.0))


    def forward(self, adjacency, input_feature):

        # 计算注意力分数
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        if self.manifold.name == 'Hyperboloid':
            attn_score = attn_score[:, 1].unsqueeze(1)

        attn_score = self.activation(attn_score)

        norm = torch.norm(input_feature, p=float('inf'), dim=1)

        # attn_score = torch.sigmoid(attn_score)
        attn_score =self.w1*attn_score + (1-self.w1)*(1-norm)+self.b

        mask = top_rank(attn_score, self.keep_ratio)
        input_feature_euc = self.manifold.proj_tan0(self.manifold.logmap0(input_feature, c=self.c), c=self.c)
        hidden = input_feature_euc[mask] * attn_score[mask].view(-1, 1)

        return hidden,attn_score

