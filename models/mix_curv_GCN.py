import torch
import torch.nn.functional as F
from torch import nn
from Model_package.mix_curv_GCN.models.base_models import GCModel
from package.package import symmetric_normalize_adj_matrix


class mix_curv_GCN(nn.Module):
    def __init__(self, args):
        super(mix_curv_GCN, self).__init__()
        self.model = GCModel(args,args.manifold0, args.manifold_array)
        self.manifold = args.manifold0
        self.c =args.c

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        self.gn = args.gn
        node_class = args.node_classes
        self.mlp1 = nn.Sequential(nn.Linear(args.dim, node_class))

    def forward(self, data):
        x, adj = data[:2]
        adj.requires_grad = False
        if self.self_loop:
            adj = adj + self.loop_att * torch.eye(x.size(0), device=x.device).unsqueeze(0)
        if self.norm_adj:
            adj = symmetric_normalize_adj_matrix(adj)

        x, adj = x.squeeze(0), adj.squeeze(0)
        adj = adj.to_sparse()
        # Encoder部分: 提取节点的表示
        emb_node,encoded_x = self.model.encode(x, adj)

        # Decoder部分: 解码得到输出
        output,h = self.model.decode(encoded_x, adj)
        logits = output

        if self.gn == 1:
            h1 = self.manifold.proj_tan0(self.manifold.logmap0(emb_node, c=self.c), c=self.c)
            logits_node = self.mlp1(h1).unsqueeze(0)
            return logits, logits_node, h
        return logits, h

