import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()

args.model = 'H2HGCN'
args.dataset = 'real'
args.readout = 'avg'

# Dataset param
args.nod_att = 'FALSE'
args.n_folds = 10
# args.log_interval = 2000
args.loss_item = 8

# Learning param
args.batch_size=1
args.save_model = 1
args.save=1

args.seed = 17
args.cuda = 0
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'


args.act  ='gelu'
args.task = 'gc'

args.self_loop = 1
args.norm_adj = 1

args.bias = 0
args.use_att = 1
args.local_agg = 0
args.manifold= 'Hyperboloid'      #PoincareBall,Euclidean''
args.skip_connections = 0

args.num_layers = 2
args.dim = 64 #嵌入维度
args.num_centroid = 20

args.lr=1e-1
args.weight_decay = 0.02
args.dropout = 0.
args.epochs = 2000
args.min_epochs = [50,30]

args.patience=30
args.c = 2
args.alpha=0.1


args.n_classes = 2
args.node_classes = 20

args.feat_dim  =40               #特征维度

args.gn = 0
args.mix = 0
args.current_time = datetime.now()
args.use_aaindex =1