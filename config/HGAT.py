import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()

args.model = 'HGAT'
args.dataset = 'real'

# Dataset param
args.nod_att = 'FALSE'
args.n_folds = 10

args.loss_item = 8       #8


# Learning param
args.batch_size=1
args.save_model = 1
args.save=1

args.seed = 17
args.cuda = 0
if args.cuda ==-1:
    args.device = 'cpu'#'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
else:
    args.device = 'cuda:' + str(args.cuda)


args.act  ='gelu'
args.task = 'gc'

args.self_loop = 1
args.norm_adj = 1

args.bias = 0
args.use_att = 1
args.use_dist_bias = 0
args.local_agg = 0

args.n_heads = 4
args.concat = 0
args.dist = 1

args.manifold= 'Hyperboloid'      #,Euclidean'PoincareBall'
args.skip_connections = 0

args.num_layers = 2
args.dim = 64 #嵌入维度
args.num_centroid = 20

args.lr=0.1
args.weight_decay = 3.913748882117371e-07
args.dropout = 0.
args.pooling_ratio = 0.5
args.epochs = 2000
args.min_epochs = [50,30]

args.patience=30
args.c = None#
args.alpha=0.1


args.feat_dim  =40              #特征维度
args.input_type = 'eucl'
args.current_time = datetime.now()

args.use_aaindex =1
