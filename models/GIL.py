import torch
import torch.nn.functional as F
from torch import nn

from Model_package.GIL.coder.decoders import DualDecoder as decoder
from Model_package.GIL.coder.encoders import GIL as encoder
from package.package import symmetric_normalize_adj_matrix


class GIL(nn.Module):
    def __init__(self, args):
        super(GIL, self).__init__()

        # 初始化 Encoder 和 Decoder
        self.encoder = encoder(args)  # 使用LGCN作为Encoder
        self.decoder = decoder(args)  # 使用LinearDecoder作为Decoder

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        self.gn = args.gn
        node_class = args.node_classes
        self.mlp1 = nn.Sequential(nn.Linear(args.dim, node_class))

    def forward(self, data):
        x, adj = data[:2]
        if self.self_loop:
            adj = adj + self.loop_att * torch.eye(x.size(0), device=x.device).unsqueeze(0)
        if self.norm_adj:
            adj = symmetric_normalize_adj_matrix(adj)
        x, adj = x.squeeze(0), adj.squeeze(0).to_sparse()
        # Encoder部分: 提取节点的表示
        emb_node, encoded_x = self.encoder.encode(x, adj)

        # Decoder部分: 解码得到输出
        output = self.decoder.decode(encoded_x, adj)
        logits = output

        if self.gn == 1:
            logits_node = self.mlp1(emb_node[1]).unsqueeze(0)
            return logits, logits_node, encoded_x

        return logits, encoded_x[1]

    def reset_parameters(self):
        self.encoder.reset_parameters()  # 重置Encoder的参数
        self.decoder.reset_parameters()  # 重置Decoder的参数


