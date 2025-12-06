import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()

args.model = 'HGCN_res'
args.dataset = 'real'

# Dataset param
args.nod_att = 'FALSE'
args.n_folds = 10




# Learning param
args.batch_size=16
args.loss_item = 1 #50//args.batch_size+1


args.save_model = 0
args.save=0

args.seed = 65
args.cuda = 0
if args.cuda ==-1:
    args.device = 'cpu'#'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
else:
    args.device = 'cuda:' + str(args.cuda)


args.act  ='relu'
args.task = 'gc'

args.self_loop = 1
args.norm_adj = 1

args.bias = 0
args.use_att = 0
args.local_agg = 0
args.manifold= 'Hyperboloid'      #PoincareBall,Euclidean''
args.skip_connections = 0

args.num_layers = 2
args.dim = 512 #嵌入维度
args.num_centroid = 20

args.pooling_ratio=0.5
args.lr=0.5
args.weight_decay = 0.00
args.dropout = 0.
args.epochs = 2000
args.min_epochs = [50,50,50]

args.patience=30
args.c = 4.489471350624781
args.alpha=0.1


args.feat_dim  =20               #特征维度

args.current_time = datetime.now()