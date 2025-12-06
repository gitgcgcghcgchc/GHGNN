import argparse
from datetime import datetime
from Model_package.mix_curv_GCN import manifolds
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()

args.loss_item =100
args.model_list = ['mix_curv_GCN']
args.model = 'HGCN'
args.dataset = 'CPTH'
args.readout = 'avg'
args.tm=60
args.self_loop = 1
args.norm_adj = 1

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
args.bias = 0
args.use_att = 0
args.local_agg = 0
args.manifold= 'H2E1S2'      #"Spherical", "Euclidean", "PoincareBall", "Hyperboloid"
args.skip_connections = 0
args.pos_weight = 0


args.num_layers = 2
args.dim = 64 #嵌入维度

args.num_centroid = 50

args.lr=1e-1
args.weight_decay = 0.02
args.dropout = 0.1
args.epochs = 2000
args.min_epochs = 60

args.patience=30
args.c = 1
args.alpha=0.1


args.n_classes = 2
args.node_classes = 20

args.feat_dim  =20               #特征维度

args.gn = 0
args.current_time = datetime.now()

# Model and optimizer
args.manifold_array = []
if args.manifold not in ["Spherical", "Euclidean", "PoincareBall", "Hyperboloid"]:
    manifold_array = []
    word = list(args.manifold)
    all_count = 0
    for i in range(0,len(word), 2):
        if word[i] == "E":
            man_name = "Euclidean"
        elif word[i] == "P":
            man_name = "PoincareBall"
        elif word[i] == "S":
            man_name = "Spherical"
        elif word[i] == "H":
            man_name = "Hyperboloid"
        else:
            raise ValueError(f"Invalid manifold name: {args.manifold}.")
        count = int(word[i+1])
        all_count +=count
        args.manifold_array.append((getattr(manifolds, man_name)(), count))
    args.total_dim = all_count*args.feat_dim
    manifold_name = "Product"
    args.manifold0 = getattr(manifolds, manifold_name)(args.manifold_array)
else:
    args.manifold0 = getattr(manifolds, args.manifold)()