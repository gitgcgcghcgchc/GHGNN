import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from Model_package.HGNN.manifold.EuclideanManifold import EuclideanManifold
from Model_package.HGNN.manifold.LorentzManifold import LorentzManifold
from Model_package.HGNN.manifold.PoincareManifold import PoincareManifold
from graphzoo import manifolds

import Model_package.HGNN.HGNN_layer as layer
from delta_tm.packages.pooling1 import global_avg_pool, SelfAttentionPooling, global_max_pool
from package.package import symmetric_normalize_adj_matrix


class HGNN(nn.Module):
    """
    Hyperbolic-GCN
    """

    def __init__(self, args):
        super(HGNN, self).__init__()


        if args.c is not None:
            self.c = torch.tensor([args.c], device=args.device)
        else:
            self.c = nn.Parameter(torch.tensor([1.], device=args.device))
        self.manifold = getattr(manifolds, args.manifold)()
        self.manifold_1 = args.manifold

        dims, acts, self.curvatures = layer.get_dim_act_curv(args)
        self.curvatures.append(self.c)

        hgc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]

            hgc_layers.append(
                layer.HGNNlayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        hidden_dim = args.dim


        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        hidden_dim = args.dim
        # self.pool = SelfAttentionPooling(hidden_dim, keep_ratio=args.pooling_ratio, activation=torch.nn.GELU(), args=args, c=self.c)

        self.distance = CentroidDistance(args, args.manifold)

        self.mlp_couple1 = nn.Sequential(
            nn.Linear(hidden_dim * 6, 128),  # 第一个全连接层
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

        # if self.manifold.name == 'Hyperboloid':
        #     o = torch.zeros_like(x1)
        #     x1 = torch.cat([o[:,:, 0:1], x1], dim=2)
        #     x2 = torch.cat([o[:,:, 0:1], x2], dim=2)
        x1, adj1 = x1.squeeze(0), adj1.squeeze(0)
        x2, adj2 = x2.squeeze(0), adj2.squeeze(0)

        x1_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x1, self.curvatures[0]), c=self.curvatures[0]),
            c=self.curvatures[0])
        x2_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x2, self.curvatures[0]), c=self.curvatures[0]),
            c=self.curvatures[0])

        if self.encode_graph:
            input1 = (x1_hyp, adj1, self.manifold_1)
            output1, _, _ = self.layers.forward(input1)

            input2 = (x2_hyp, adj2, self.manifold_1)
            output2, _, _ = self.layers.forward(input2)
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

        logits3 = self.mlp_couple1(torch.cat((h1, h2,h1-h2), dim=0))

        logits = logits3, logits3, logits3
        h = h1, h2

        return logits, h



class ResNetBlockHGNN(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, c_in, c_out, act, dropout, bias, use_att, local_agg):
        super(ResNetBlockHGNN, self).__init__()

        self.conv1 = layer.HGNNlayer(
            manifold, in_dim, out_dim, c_in, c_out, dropout, act, bias, use_att, local_agg
        )
        self.conv2 = layer.HGNNlayer(
            manifold, out_dim, out_dim, c_out, c_out, dropout, act, bias, use_att, local_agg
        )
        self.shortcut = nn.Sequential()

        # If input and output dimensions do not match, add a linear transformation to the shortcut
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )

    def forward(self,input):
        x, adj, manifold_type =input
        out, _, _ = self.conv1((x, adj, manifold_type))
        out, _, _ = self.conv2((out, adj, manifold_type))
        return out + self.shortcut(x), adj, manifold_type

class HGNN_res(nn.Module):
    """
    带有残差块的 Hyperbolic-GCN
    """

    def __init__(self, args):
        super(HGNN_res, self).__init__()

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to('cuda:' + str(args.cuda))
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, args.manifold)()
        self.manifold_1 = args.manifold

        dims, acts, self.curvatures = layer.get_dim_act_curv(args)
        self.curvatures.append(self.c)

        # 使用带残差块的 HGNN 层
        hgc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]

            hgc_layers.append(
                ResNetBlockHGNN(
                    self.manifold, in_dim, out_dim, c_in, c_out, act, args.dropout, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        hidden_dim = args.dim

        num_classes = args.n_classes
        num_centroid = args.num_centroid

        self.distance = CentroidDistance(args, args.manifold)
        self.mlp = nn.Sequential(nn.Linear(num_centroid, num_classes))

        self.gn = args.gn

        node_class = args.node_classes
        self.mlp1 = nn.Sequential(nn.Linear(args.dim, node_class))

    def forward(self, x, adj, graph_indicator):

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        if self.encode_graph:
            input = (x_hyp, adj, self.manifold_1)

            output, _, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x_hyp)

        # readout = mean_pool(output, graph_indicator)
        readout, _ = self.distance(output, graph_indicator)

        h = self.manifold.proj_tan0(self.manifold.logmap0(readout, c=self.c), c=self.c)

        logits = F.softmax(self.mlp(h), dim=0).unsqueeze(0)

        if self.gn == 1:
            h1 = self.manifold.proj_tan0(self.manifold.logmap0(output, c=self.c), c=self.c)
            logits_node = F.softmax(self.mlp1(h1), dim=0).unsqueeze(0)
            return logits, logits_node, h
        return logits, h

class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids.
    All nodes belong to the same graph.
    """
    def __init__(self, args, manifold):
        super(CentroidDistance, self).__init__()
        self.args = args
        self.manifold = manifold

        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, args.dim,
            sparse=False,
            scale_grad_by_freq=False,
        )

        if args.manifold == 'Hyperboloid':
            self.manifold_1 = LorentzManifold(args)
        elif args.manifold == 'PoincareBall':
            self.manifold_1 = PoincareManifold(args)
        elif args.embed_manifold == 'euclidean':
            self.manifold_1 = EuclideanManifold(args)
        self.manifold_1.init_embed(embed=self.centroid_embedding)

    def forward(self, node_repr):
        """
        Args:
            node_repr: [node_num, embed_size] -- 所有节点的嵌入矩阵

        Returns:
            graph_centroid_dist: [1, num_centroid] -- 所有节点到各个质心的平均距离
            node_centroid_dist: [node_num, num_centroid] -- 每个节点到所有质心的距离
        """
        node_num = node_repr.size(0)

        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr_expanded = node_repr.unsqueeze(1).expand(
            -1,
            self.args.num_centroid,
            -1
        ).contiguous().view(-1, self.args.dim)

        # 获取质心嵌入
        centroid_repr = self.centroid_embedding(th.arange(self.args.num_centroid).to(node_repr.device))

        centroid_repr = centroid_repr.unsqueeze(0).expand(
            node_num,
            -1,
            -1
        ).contiguous().view(-1, self.args.dim)

        # 计算节点与质心之间的距离
        node_centroid_dist = self.manifold_1.distance(node_repr_expanded, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(node_num, self.args.num_centroid)

        # 计算所有节点到各个质心的平均距离
        graph_centroid_dist = node_centroid_dist.mean(dim=0, keepdim=True)

        return graph_centroid_dist, node_centroid_dist
