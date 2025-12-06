import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

import torch
import optuna

from dataset.graph_data_reader_couple import get_data
from delta_tm.train_3_1 import train_model

from config.GHG import args
from delta_tm.packages.random import seed_everything

if __name__ == '__main__':
    seed_everything(args.seed)
    # 在训练脚本开头添加
    # torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 开启Tensor Core加速


    print('Target model:', args.model)
    print('Target dataset:', args.dataset)
    print('Using device in train process:', args.device)
    print('-' * 50)

    # Build graph data reader:
    datareader,splits = get_data(args.dataset, args.seed, args.n_folds)

    print('-' * 25)


    def objective(trial):
        # args.loss_item = trial.suggest_int("loss_item", 1, 50)
        args.dim = trial.suggest_categorical("dim", [64,128,256])#8,16,32,
        args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-9, 1e-6)
        # args.dropout = trial.suggest_uniform("dropout_rate", 0., 0.5)
        # args.pooling_ratio = trial.suggest_uniform("pooling_ratio", 0.1, 0.9)
        # args.c = trial.suggest_uniform("c", 0.1, 10)
        # args.seed = trial.suggest_int("seed", 0, 1000)
        # args.patience = trial.suggest_int("patience", 0, 50)
        # args.n_heads = 2**trial.suggest_int("n_heads", 1, 3)
        # args.num_layers = trial.suggest_int("num_layers", 2, 3)
        args.lr = trial.suggest_float("lr", 1e-2, 5e-1, log=True)
        # args.use_dist_bias = trial.suggest_categorical("use_dist_bias", [0,1])


        val_mean, test_mean = train_model(args,  datareader, splits)
        print('test_mean is:',test_mean)
        return test_mean

    # 创建 study 并运行优化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params, study.best_value)
