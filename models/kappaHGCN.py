import networkx
import torch
import torch.nn.functional as F
from torch import nn

import Model_package.kappaHGCN.coder.encoders as encoders
import Model_package.kappaHGCN.manifolds as manifolds
from Model_package.kappaHGCN.coder.decoders import model2decoder
from Model_package.kappaHGCN.layers.layers import FermiDiracDecoder
from Model_package.kappaHGCN.utils.eval_utils import acc_f1
from Model_package.kappaHGCN.utils.negative_sampling import negative_sampling
from package.package import symmetric_normalize_adj_matrix


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        self.encoder = getattr(encoders, args.model_list[0])(args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class GCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(GCModel, self).__init__(args)
        self.decoder = model2decoder[args.model_list[0]](args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            1
            # self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.edge_false = None
        self.filtered_edges = None
        self.args = args

    def lp_decode(self, h, edge_index):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_i = torch.nn.functional.embedding(edge_i, h)
        x_j = torch.nn.functional.embedding(edge_j, h)
        sqdist = self.manifold.sqdist(x_i, x_j, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def filter_leaf(self, edges_true, orc, threshold, dataset):
        if self.filtered_edges is not None:
            return self.filtered_edges
        edge_i = list(edges_true[0].cpu().numpy())
        edge_j = list(edges_true[1].cpu().numpy())

        DG = networkx.Graph()
        for edge in edges_true.transpose(1, 0).cpu().numpy():
            if edge[0] != edge[1]:
                DG.add_edge(edge[0], edge[1])

        filtered = []
        for ix in range(edges_true.size(1)):
            node_i = edge_i[ix]
            node_j = edge_j[ix]

            if DG.has_node(node_i) and DG.has_node(node_j):  # Non-isolated nodes
                if DG.degree(node_i) == 1 or DG.degree(node_j) == 1:  # If it is a leaf node, it will not be considered.
                    filtered.append(False)
                else:
                    if orc[ix] > threshold:
                        filtered.append(True)  # If not a leaf node and the value of orc satisfies the condition
                    else:  # Filter out nodes with smaller orc values
                        filtered.append(False)
            else:  # Isolated nodes
                filtered.append(False)
        self.filtered_edges = filtered_edges = edges_true[:, filtered]
        return filtered_edges

    def compute_metrics_orc(self, embeddings, data, split, **kargs):
        assert self.args.agg_type in ['curv', 'attcurv']
        orc = data['adj_train_norm'][1].squeeze()
        threshold = kargs['threshold']
        dataset = kargs['dataset']
        edges_true = data['edges_true']
        filtered_edges = self.filter_leaf(edges_true, orc, threshold, dataset)

        if filtered_edges.size(1) == 0:
            return {'loss': 0}

        if not self.edge_false:
            self.edges_false = negative_sampling(edges_true, num_nodes=embeddings.shape[0],
                                                 num_neg_samples=filtered_edges.size(1))

        pos_scores = self.lp_decode(embeddings, filtered_edges)
        neg_scores = self.lp_decode(embeddings, self.edges_false)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        metrics = {'loss': loss}
        return metrics

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output.unsqueeze(0)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)

        # curv = curv_acc(output, data['labels'][idx], data['node_curvature'])

        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class kappaHGCN(nn.Module):
    def __init__(self, args):
        super(kappaHGCN, self).__init__()
        self.args = args
        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.c])
            self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        self.encoder = getattr(encoders, args.model_list[0])(args)
        self.model = GCModel(self.args)

        self.self_loop = args.self_loop
        self.loop_att = torch.tensor(10., requires_grad=True)
        self.norm_adj = args.norm_adj

        self.gn = args.gn
        node_class = args.node_classes
        self.mlp1 = nn.Sequential(nn.Linear(args.dim, node_class))


    def forward(self,data):
        x, adj = data[:2]

        x = x.squeeze(0)
        node_embeddings = self.model.encode(x, adj)
        graph_embddings = torch.mean(node_embeddings, dim=0).squeeze()
        logits = self.model.decode(graph_embddings.unsqueeze(0), adj)

        if self.gn == 1:
            h1 = self.manifold.proj_tan0(self.manifold.logmap0(node_embeddings, c=self.c), c=self.c)
            logits_node = self.mlp1(h1).unsqueeze(0)
            return logits, logits_node, graph_embddings
        return logits,graph_embddings