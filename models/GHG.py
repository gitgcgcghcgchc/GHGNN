
import graphzoo.layers.hyp_layers as hyp_layers
import numpy as np
import torch
import torch.nn as nn
from graphzoo import manifolds

from delta_tm.packages.attention import GraphSelfAttention
from delta_tm.packages.pooling1 import global_avg_pool, global_max_pool, SelfAttentionPooling
from delta_tm.packages.package import symmetric_normalize_adj_matrix


class ResNetBlock(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, c_in, c_out, act, dropout, bias, use_att,use_att1, local_agg, heads, use_dist_bias, use_res):
        super(ResNetBlock, self).__init__()
        self.manifold = manifold
        self.c = c_in
        self.conv1 = hyp_layers.HyperbolicGraphConvolution(
            manifold, in_dim, out_dim, c_in, c_out, dropout, act, bias, use_att, local_agg)
        # self.conv2 = hyp_layers.HyperbolicGraphConvolution(
        #     manifold, out_dim, out_dim, c_out, c_out, dropout, act, bias, use_att, local_agg)
        self.shortcut = nn.Sequential()

        # If input and output dimensions do not match, add a linear transformation to the shortcut
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                hyp_layers.HypLinear(manifold, in_dim, out_dim, c_in, dropout, 0),

            )
        self.use_att = use_att
        self.use_res = use_res
        self.use_att1 = use_att1
        self.k = nn.Parameter(torch.Tensor([1.]))
        if self.use_att1:
            self.att1 = GraphSelfAttention(manifold, out_dim, out_dim, c_in, heads=heads, dropout=dropout, use_dist_bias=use_dist_bias)

    def forward(self, input):
        x, adj, dist = input

        out1,adj = self.conv1((x, adj))
        if self.use_att:
            out1, attn1 = self.att1(out1, adj, dist)

        if self.use_res:
            return self.manifold.mobius_add_matrix(out1,self.k*self.shortcut(x), self.c), adj, dist
        else:
            return out1, adj, dist

class GHG(nn.Module):
    """
    Hyperbolic-GCN with ResNet Blocks
    """

    def __init__(self, args):
        super(GHG, self).__init__()

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to('cuda:' + str(args.cuda))
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, args.manifold)()

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        self.dims = dims

        # Using ResNet Blocks instead of simple HGC layers
        hgc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                ResNetBlock(
                    self.manifold, in_dim, out_dim, c_in, c_out, act, args.dropout, args.bias, args.use_att,args.use_att1,
                    args.local_agg, args.n_heads, args.use_dist_bias, args.use_res
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        hidden_dim = args.dim

        self.use_pool =args.use_pool
        if args.use_pool:
            self.pool  = SelfAttentionPooling(hidden_dim, keep_ratio=args.pooling_ratio,activation=torch.nn.GELU(),args=args, c=self.c)

        self.mlp_couple1 = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),  # 第一个全连接层
            nn.GELU(),  # 激活函数
            nn.Dropout(args.dropout),  # Dropout层
            nn.Linear(hidden_dim, 1)  # 输出层
        )

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=False)
        self.norm_adj = args.norm_adj

        self.use_aaindex = args.use_aaindex

    def forward(self, data):

        x1, x2, adj1, adj2 = data[:4]
        dist1, dist2 = data[10:12]
        aaindex_feature1, aaindex_feature2 = data[12:14]
        # position_emb1, position_emb2 = data[12:14]



        if self.use_aaindex:
            x1 = torch.cat([x1, aaindex_feature1], dim=2)
            x2 = torch.cat([x2, aaindex_feature2], dim=2)

        if self.self_loop:
            adj1 = adj1 + self.loop_att * torch.eye(x1.size(0), device=x1.device).unsqueeze(0)
            adj2 = adj2 + self.loop_att * torch.eye(x2.size(0), device=x2.device).unsqueeze(0)
        if self.norm_adj:
            adj1 = symmetric_normalize_adj_matrix(adj1)
            adj2 = symmetric_normalize_adj_matrix(adj2)

        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x1)
            x1 = torch.cat([o[:,:, 0:1], x1], dim=2)
            x2 = torch.cat([o[:,:, 0:1], x2], dim=2)

        x1, adj1 = x1.squeeze(0), adj1.squeeze(0)
        x2, adj2 = x2.squeeze(0), adj2.squeeze(0)


        x1_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x1, self.curvatures[0]), c=self.curvatures[0]),
            c=self.curvatures[0])
        x2_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x2, self.curvatures[0]), c=self.curvatures[0]),
            c=self.curvatures[0])


        if self.encode_graph:
            input1 = (x1_hyp, adj1, dist1)
            output1, _, _ = self.layers.forward(input1)

            input2 = (x2_hyp, adj2, dist2)
            output2, _, _ = self.layers.forward(input2)
        else:
            output1 = self.layers.forward(x1_hyp)
            output2 = self.layers.forward(x2_hyp)
        if self.use_pool ==1:
            pool1,pool_score1 = self.pool(adj1, output1)
            readout1 = torch.cat((global_avg_pool(pool1),global_max_pool(pool1)), dim=1).squeeze()
            # readout1 = torch.cat((self.manifold.mobius_mean(pool1, self.c), global_max_pool(pool1)), dim=1).squeeze()

            pool2,pool_score2 = self.pool(adj2, output2)
            readout2 = torch.cat((global_avg_pool(pool2), global_max_pool(pool2)), dim=1).squeeze()
            # readout2 = torch.cat((self.manifold.mobius_mean(pool2, self.c), global_max_pool(pool2)), dim=1).squeeze()
        else:
            readout1 = torch.cat((global_avg_pool(output1),global_max_pool(output1)), dim=1).squeeze()
            readout2 = torch.cat((global_avg_pool(output2), global_max_pool(output2)), dim=1).squeeze()
            pool_score1,pool_score2 = np.zeros((1,readout1.shape[0])), np.zeros((1,readout2.shape[0]))

        # h1 = self.manifold.proj_tan0(self.manifold.logmap0(readout1.unsqueeze(0), c=self.c), c=self.c).squeeze()
        # h2 = self.manifold.proj_tan0(self.manifold.logmap0(readout2.unsqueeze(0), c=self.c), c=self.c).squeeze()
        h1,h2=readout1,readout2
        logits3 = self.mlp_couple1(torch.cat((h1, h2, h1-h2), dim=0))
        logits = logits3, logits3, logits3
        h = h1, h2


        return logits, h, output1, output2, pool_score1, pool_score2




