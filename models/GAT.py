import os
import sys

from delta_tm.packages.pooling import SelfAttentionPooling, global_avg_pool, global_max_pool

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_attention_layer import GraphAttentionLayer
from package.package import symmetric_normalize_adj_matrix


"""
Base paper: https://arxiv.org/abs/1710.10903
"""

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()

        self.n_layer = args.num_layers
        self.dropout = args.dropout

        
        # Graph attention layer
        self.graph_attention_layers = []
        for i in range(self.n_layer):
          self.graph_attention_layers.append(GraphAttentionLayer(args.feat_dim, args.dim, args.dropout, args.device))
        self.layers =nn.Sequential(*self.graph_attention_layers)
        self.pool = SelfAttentionPooling(2*args.dim, keep_ratio=args.pooling_ratio, activation=torch.nn.GELU())

        self.mlp_couple1 = nn.Sequential(
            nn.Linear(args.dim * 12, 128),  # 第一个全连接层
            nn.GELU(),  # 激活函数
            nn.Dropout(args.dropout),  # Dropout层
            nn.Linear(128, 1)  # 输出层
        )
        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj
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

        # Dropout        
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # Graph attention layer
        x1 = torch.cat([F.relu(att(x1, adj1)) for att in self.graph_attention_layers], dim=2).squeeze()
        x2 = torch.cat([F.relu(att(x2, adj2)) for att in self.graph_attention_layers], dim=2).squeeze()


        # Readout
        pool1 = self.pool(adj1.squeeze(), x1)
        readout1 = torch.cat((global_avg_pool(pool1).squeeze(0), global_max_pool(pool1).squeeze(0)), dim=1).squeeze()
        pool2 = self.pool(adj2.squeeze(), x2)
        readout2 = torch.cat((global_avg_pool(pool2).squeeze(0), global_max_pool(pool2).squeeze(0)), dim=1).squeeze()

        logits3 = self.mlp_couple1(torch.cat((readout1, readout2, readout1 - readout2), dim=0))

        logits = logits3, logits3, logits3
        h = readout1, readout2

        return logits,h
