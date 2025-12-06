import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import geoopt.manifolds.stereographic.math as pmath
from Model_package.GIL.layers.layers import GCNConv, GATConv, HFusion, EFusion
from Model_package.GIL.layers.layers import remove_self_loops, add_self_loops, softmax, MessagePassing, glorot, zeros

from  graphzoo.layers.hyp_layers import HypLinear, HypAgg, HypAct


class HGATLayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, act, args):
        super(HGATLayer, self).__init__()
        self.conv = HGATConv(manifold, in_features, out_features, args.n_heads, args.concat, args.alpha, args.dropout,
                             args.bias, act, c, dist=0)

    def forward(self, input):
        x = input[0]
        adj = input[1]
        "hyper forward"
        input_h = x, adj
        x = self.conv(input_h)
        return x, adj


class GILayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, act, args):
        super(GILayer, self).__init__()
        self.conv = HGATConv(manifold, in_features, out_features, args.n_heads, args.concat, args.alpha, args.dropout,
                             args.bias, act, atten=args.atten, dist=args.dist)
        self.conv_e = GATConv(in_features, out_features, args.n_heads, args.concat, args.alpha,
                              args.dropout, args.bias, act)

        '''feature fusion'''
        self.h_fusion = HFusion(c, args.drop_e)
        self.e_fusion = EFusion(c, args.drop_h)

    def forward(self, input):
        x, x_e = input[0]
        adj = input[1]
        "hyper forward"
        input_h = x, adj
        x = self.conv(input_h)

        "eucl forward"
        input_e = x_e, adj
        x_e, _ = self.conv_e(input_e)

        "feature fusion"
        x = self.h_fusion(x, x_e)
        x_e = self.e_fusion(x, x_e)

        return (x, x_e), adj


class HGATConv(MessagePassing):
    def __init__(self,
                 manifold,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None,
                 c= torch.tensor(1.0),
                 atten=True,
                 dist=True):
        super(HGATConv, self).__init__('add')

        self.manifold = manifold
        self.c = c
        self.concat = concat
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.in_channels = in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        self.dist = dist
        self.atten = atten

        self.hy_linear = HypLinear(manifold, in_channels, heads * self.out_channels, self.c, dropout, bias)
        self.att = Parameter(torch.Tensor( heads,1, 2 * self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.hy_linear.reset_parameters()

    def forward(self, input):
        decoder = 0
        if len(input) == 3:
            x, adj, decoder = input
        else:
            x, adj = input
        x = self.hy_linear.forward(x)
        edge_index = adj._indices()

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        log_x = pmath.logmap0(x, k=(1./self.c))  # Get log(x) as input to GCN

        log_x = log_x.view(self.heads, -1, self.out_channels)
        if decoder==0:
            out = self.propagate(edge_index, x=log_x, size=(x.size(0), x.size(0)), original_x=x, num_nodes=x.size(0))
        else:
            out = log_x.squeeze(0)
        out = self.manifold.proj_tan0(out, c=self.c)

        out = self.act(out)
        out = self.manifold.proj_tan0(out, c=self.c)

        return self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

    def message(self, edge_index_i, x_i, x_j, num_nodes, original_x_i, original_x_j):
        if self.atten:

            # x_i 和 x_j 的形状应为 [num_edges, heads, out_channels]
            assert x_i.shape == x_j.shape, "x_i and x_j must have the same shape"

            # 拼接 x_i 和 x_j: [num_edges, heads, 2 * out_channels]
            x_cat = torch.cat([x_i, x_j], dim=-1)

            # 计算注意力得分 alpha: [num_edges, heads]
            alpha = torch.matmul(x_cat, self.att.transpose(1, 2)).squeeze(-1)  # (1)

            if self.dist:
                # 计算双曲距离: [num_edges, 1]

                dist = pmath.dist(original_x_i, original_x_j, k=1.0/self.c)

                dist = softmax(dist, edge_index_i, num_nodes=num_nodes).unsqueeze(0)  # (2)


                # 将距离与 alpha 相乘: [num_edges, heads]
                alpha = alpha * dist  # 广播机制自动扩展 dist 到 [num_edges, heads]

            # 激活和归一化
            alpha = F.leaky_relu(alpha, self.negative_slope)

            alpha = softmax(alpha.transpose(0, 1), edge_index_i, num_nodes=num_nodes).transpose(0, 1)  # (3)

            # 训练时随机失活
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            # 应用注意力权重: [num_edges, heads, out_channels]

            return x_j * alpha.unsqueeze(-1)  # (4)
        else:

            return x_j

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=0)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
