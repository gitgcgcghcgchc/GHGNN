import os
import sys

import torch

from delta_tm.packages.pooling import SelfAttentionPooling
from delta_tm.packages.pooling1 import global_avg_pool, global_max_pool
from package.package import symmetric_normalize_adj_matrix

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from layers.graph_resnet_layer import GraphResNetLayer
from layers.graph_convolution_layer import GraphConvolutionLayer

class GraphResNet(nn.Module):
    def __init__(self, args):
        super(GraphResNet, self).__init__()
        
        self.n_layer = args.num_layers
        self.dropout = args.dropout
        
        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(args.feat_dim, args.dim, args.device)
        
        # Graph resnet layer
        self.graph_resnet_layers = []
        for i in range(self.n_layer):
             self.graph_resnet_layers.append(GraphResNetLayer(args.dim, args.dim, args.device))
        self.layers = nn.Sequential(*self.graph_resnet_layers)
        self.pool = SelfAttentionPooling(args.dim, keep_ratio=args.pooling_ratio)

        # self.mlp = nn.Sequential(nn.Linear(args.hidden_dim * 2, 1), )
        self.mlp_couple = nn.Sequential(nn.Linear(args.dim * 8, 1))

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

    def forward(self, data):
        x1, x2, adj1, adj2 = data[:4]

        if self.self_loop:
            adj1 = adj1 + self.loop_att * torch.eye(x1.size(0), device=x1.device).unsqueeze(0)
            adj2 = adj2 + self.loop_att * torch.eye(x2.size(0), device=x2.device).unsqueeze(0)
        if self.norm_adj:
            adj1 = symmetric_normalize_adj_matrix(adj1)
            adj2 = symmetric_normalize_adj_matrix(adj2)

        # Graph convolution layer
        x1 = F.relu(self.graph_convolution_layer(x1, adj1))
        x2 = F.relu(self.graph_convolution_layer(x2, adj2))
        # Dropout
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        for i in range(self.n_layer):
           # Graph resnet layer
           x1 = F.relu(self.graph_resnet_layers[i](x1, adj1))
           x2 = F.relu(self.graph_resnet_layers[i](x2, adj2))
           # Dropout
           if i != self.n_layer - 1:
                x1 = F.dropout(x1, p=self.dropout, training=self.training)
                x2 = F.dropout(x2, p=self.dropout, training=self.training)


        pool1 = self.pool(adj1, x1).squeeze()
        readout1 = torch.cat((global_avg_pool(pool1), global_max_pool(pool1)), dim=1).squeeze()
        pool2 = self.pool(adj2, x2).squeeze()
        readout2 = torch.cat((global_avg_pool(pool2), global_max_pool(pool2)), dim=1).squeeze()

        # Fully-connected layer
        # logits1 = self.mlp(readout1)
        # logits2 = self.mlp(readout2)
        logits3 = self.mlp_couple(torch.cat((readout1, readout2, readout1 + readout2, readout1 - readout2), dim=0))

        logits = logits3, logits3, logits3
        h = readout1, readout2

        return logits, h
        

            
            