import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()
args.loss_item = 100
args.model_list = ['kappaHGCN']
args.dataset = 'CPTH'
args.readout = 'avg'
args.tm=60


# Dataset param
args.nod_att = 'FALSE'
args.n_folds = 10
args.log_interval = 2000

# Learning param
args.batch_size=1
args.save_model = 0
args.save=1

args.seed = 65

args.cuda = 0
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

args.act  ='relu'
args.task = 'gc'

args.self_loop = 1
args.norm_adj = 1

args.bias = 0
args.use_att = 0
args.local_agg = 0
args.manifold= 'PoincareBall'      #Hyperboloid,Euclidean
args.skip_connections = 0

args.num_layers = 2
args.dim = 64 #嵌入维度
args.num_centroid = 50

args.lr=1e-1
args.weight_decay = 0.02
args.dropout = 0.1
args.epochs = 2000
args.min_epochs = 10

args.patience=30
args.c = 1
args.alpha=0.1


args.n_classes = 2
args.node_classes = 20
args.feat_dim  =20               #特征维度

args.gn = 0

args.agg_type= 'attcurv'     #attsparse, attdense, attpyg, deg
args.position = 'origin'         #'origin or local'
args.pos_weight = 0
args.r = 2
args.t = 1
args.current_time = datetime.now()