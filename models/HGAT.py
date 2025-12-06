import torch
import torch.nn.functional as F
from torch import nn
from Model_package.GIL import manifolds
from Model_package.GIL.coder.decoders import LinearDecoder as decoder
from Model_package.GIL.coder.encoders import HGAT as encoder
from delta_tm.packages.pooling1 import SelfAttentionPooling, global_avg_pool, global_max_pool
from package.package import symmetric_normalize_adj_matrix


class HGAT(nn.Module):
    def __init__(self, args):
        super(HGAT, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to('cuda:' + str(args.cuda))
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        # 初始化 Encoder 和 Decoder
        self.encoder = encoder(args)  # 使用HGAT作为Encoder
        self.manifold = getattr(manifolds, args.manifold)()

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        # self.pool = SelfAttentionPooling(args.dim, keep_ratio=args.pooling_ratio, activation=torch.nn.GELU())

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

        if self.manifold.name == 'Hyperboloid':
            o1 = torch.zeros_like(x1)
            o2 = torch.zeros_like(x2)
            x1 = torch.cat([o1[:, :, 0:1], x1], dim=2)
            x2 = torch.cat([o2[:, :, 0:1], x2], dim=2)

        x1, adj1 = x1.squeeze(0), adj1.squeeze(0).to_sparse()
        x2, adj2 = x2.squeeze(0), adj2.squeeze(0).to_sparse()

        # Encoder部分: 提取节点的表示
        emb_node1, encoded_x1_hyp = self.encoder.encode(x1, adj1)
        emb_node2, encoded_x2_hyp = self.encoder.encode(x2, adj2)

        # Decoder部分: 解码得到输出
        # pool1 = self.pool(adj1, emb_node1)
        readout1 = torch.cat((global_avg_pool(emb_node1),global_max_pool(emb_node1)), dim=1).squeeze()
        # pool2 = self.pool(adj2, emb_node2)
        readout2 = torch.cat((global_avg_pool(emb_node2), global_max_pool(emb_node2)), dim=1).squeeze()


        h1 = self.manifold.proj_tan0(self.manifold.logmap0(readout1.unsqueeze(0), c=self.c), c=self.c).squeeze()
        h2 = self.manifold.proj_tan0(self.manifold.logmap0(readout2.unsqueeze(0), c=self.c), c=self.c).squeeze()


        logits3 = self.mlp_couple1(torch.cat((h1, h2, h1-h2), dim=0))

        logits = logits3, logits3, logits3
        h =h1,h2

        return logits,h

    def reset_parameters(self):
        self.encoder.reset_parameters()  # 重置Encoder的参数
        self.decoder.reset_parameters()  # 重置Decoder的参数


