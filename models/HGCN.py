
import graphzoo.layers.hyp_layers as hyp_layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphzoo import manifolds

from delta_tm.packages.pooling1 import SelfAttentionPooling, global_avg_pool, global_max_pool
from delta_tm.packages.package import symmetric_normalize_adj_matrix


class HGCN(nn.Module):
    """
    Hyperbolic-GCN
    """

    def __init__(self, args):
        super(HGCN, self).__init__()

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to('cuda:' + str(args.cuda))
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))


        self.manifold = getattr(manifolds, args.manifold)()


        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)

        hgc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]

            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        hidden_dim = args.dim
        # self.pool = SelfAttentionPooling(hidden_dim, keep_ratio=args.pooling_ratio, activation=torch.nn.GELU(), args=args, c=self.c)

        self.mlp_couple1 = nn.Sequential(
            nn.Linear(hidden_dim * 6, 128),  # 第一个全连接层
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
            adj2= adj2 + self.loop_att * torch.eye(x2.size(0), device=x2.device).unsqueeze(0)
        if self.norm_adj:
            adj1 = symmetric_normalize_adj_matrix(adj1)
            adj2 = symmetric_normalize_adj_matrix(adj2)

        if self.manifold.name == 'Hyperboloid':
            o1 = torch.zeros_like(x1)
            o2 = torch.zeros_like(x2)
            x1 = torch.cat([o1[:,:, 0:1], x1], dim=2)
            x2 = torch.cat([o2[:,:, 0:1], x2], dim=2)

        x1,adj1 = x1.squeeze(0), adj1.squeeze(0)
        x2,adj2 = x2.squeeze(0), adj2.squeeze(0)

        x1_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x1, self.curvatures[0]),c=self.curvatures[0]), c=self.curvatures[0])
        x2_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x2, self.curvatures[0]),c=self.curvatures[0]), c=self.curvatures[0])

        if self.encode_graph:
            input1 = (x1_hyp, adj1)
            output1, _ = self.layers.forward(input1)

            input2 = (x2_hyp, adj2)
            output2, _ = self.layers.forward(input2)
        else:
            output1 = self.layers.forward(x1_hyp)
            output2 = self.layers.forward(x2_hyp)

        # pool1 = self.pool(adj1, output1)
        readout1 = torch.cat((global_avg_pool(output1),global_max_pool(output1)), dim=1).squeeze()
        # readout1 = global_avg_pool(pool1).squeeze()
        # pool2 = self.pool(adj2, output2)
        readout2 = torch.cat((global_avg_pool(output2), global_max_pool(output2)), dim=1).squeeze()
        # readout2 = global_avg_pool(pool2).squeeze()

        h1 = self.manifold.proj_tan0(self.manifold.logmap0(readout1.unsqueeze(0), c=self.c), c=self.c).squeeze()
        h2 = self.manifold.proj_tan0(self.manifold.logmap0(readout2.unsqueeze(0), c=self.c), c=self.c).squeeze()

        logits3 = self.mlp_couple1(torch.cat((h1, h2, h1-h2), dim=0))

        logits = logits3, logits3, logits3
        h = h1, h2


        return logits, h,_,_,_,_








