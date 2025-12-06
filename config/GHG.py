import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()

args.model = 'GHG'
args.dataset = 'StructΔTm801'

# Dataset param
args.nod_att = 'FALSE'
args.n_folds = 10

args.loss_item = 128       #8


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

args.bias = 1
args.use_att = 1
args.use_att1 =1
args.use_res = 1
args.use_pool = 1
args.use_dist_bias = 1
args.local_agg = 0


args.n_heads = 4
args.dist = 1

args.manifold= 'PoincareBall'      #,'Hyperboloid'Euclidean


args.num_layers = 2
args.dim = 128 #嵌入维度
args.num_centroid = 20

args.lr=0.01
args.weight_decay = 3.0e-07
args.dropout = 0.
args.pooling_ratio = 0.5
args.epochs = 500
args.min_epochs = [10,10]

args.patience=10
args.c = None#



args.feat_dim  =40              #特征维度
args.input_type = 'eucl'
args.current_time = datetime.now()

args.use_aaindex =1
