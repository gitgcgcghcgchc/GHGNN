import os
import sys

import torch
from graphzoo.layers import hyp_layers

from delta_tm.packages.pooling import  SelfAttentionPooling
from delta_tm.packages.pooling import global_avg_pool, global_max_pool
from package.package import symmetric_normalize_adj_matrix

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from graphzoo.layers.layers import GraphConvolution
from package.readouts.basic_readout import readout_function

"""
Base paper: https://arxiv.org/abs/1609.02907
"""
class ResNetBlock(nn.Module):
    def __init__(self,in_dim, out_dim,act, dropout, bias):
        super(ResNetBlock, self).__init__()

        self.conv1 = GraphConvolution(in_dim, out_dim, dropout, act, bias)

        self.shortcut = nn.Sequential()

        # If input and output dimensions do not match, add a linear transformation to the shortcut
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                torch.nn.Linear(in_dim, out_dim,bias=bias),
                nn.LayerNorm(out_dim)
            )

    def forward(self, input):
        x, adj = input

        out,adj = self.conv1((x, adj))
        return out + self.shortcut(x), adj

class GCN_res(nn.Module):
    def __init__(self, args):
        super(GCN_res, self).__init__()
        
        self.n_layer = args.num_layers
        self.dropout = args.dropout



        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        act = getattr(F, args.act)
        # Graph convolution layer
        gc_layers = []
        for i in range(self.n_layer):
           if i == 0:
             gc_layers.append(ResNetBlock(args.feat_dim, args.dim,act, args.dropout,  args.bias))
           else:
             gc_layers.append(ResNetBlock(args.dim, args.dim,act, args.dropout,args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.pool = SelfAttentionPooling(args.dim, keep_ratio=args.pooling_ratio, activation=torch.nn.GELU())

        self.mlp_couple1 = nn.Sequential(
            nn.Linear(args.dim * 6, 128),  # 第一个全连接层
            nn.GELU(),  # 激活函数
            nn.Dropout(args.dropout),  # Dropout层
            nn.Linear(128, 1)  # 输出层
        )

        self.use_aaindex = args.use_aaindex

    def forward(self, data):

        x1, x2, adj1, adj2 = data[:4]
        aaindex_feature1, aaindex_feature2 = data[12:14]

        if self.use_aaindex:
            x1 = torch.cat([x1, aaindex_feature1], dim=2)
            x2 = torch.cat([x2, aaindex_feature2], dim=2)

        if self.self_loop:
            adj1 = adj1 + self.loop_att * torch.eye(x1.size(0), device=x1.device).unsqueeze(0)
            adj2 = adj2 + self.loop_att * torch.eye(x2.size(0), device=x2.device).unsqueeze(0)
        if self.norm_adj:
            adj1 = symmetric_normalize_adj_matrix(adj1)
            adj2 = symmetric_normalize_adj_matrix(adj2)

        x1, adj1 = x1.squeeze(0), adj1.squeeze(0)
        x2, adj2 = x2.squeeze(0), adj2.squeeze(0)

        input1 = (x1, adj1)

        output1, _ = self.layers.forward(input1)

        input2 = (x2, adj2)
        output2, _ = self.layers.forward(input2)



        # Readout
        pool1 = self.pool(adj1, output1)
        readout1 = torch.cat((global_avg_pool(pool1).squeeze(0),global_max_pool(pool1).squeeze(0)), dim=1).squeeze()
        pool2 = self.pool(adj2, output2)
        readout2 = torch.cat((global_avg_pool(pool2).squeeze(0), global_max_pool(pool2).squeeze(0)), dim=1).squeeze()
        logits3 = self.mlp_couple1(torch.cat((readout1, readout2, readout1-readout2), dim=0))

        logits = logits3,logits3,logits3
        h = readout1, readout2
        return logits, h
            
            