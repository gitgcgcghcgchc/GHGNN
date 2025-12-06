#训练batch_size==1
import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path
sys.path.append(current_dir)
import torch
from dataset.graph_data_reader_couple import get_data
# from delta_tm.test1 import train_model
from delta_tm.val import val_model
from config.GHG import args


if __name__ == '__main__':
    # 在训练脚本开头添加
    # torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 开启Tensor Core加速


    print('Target model:', args.model)
    print( 'Target dataset:', args.dataset)
    print('Using device in train process:', args.device)
    print('-' * 50)

    # Build graph data reader:
    datareader,splits = get_data(args.dataset, args.seed, args.n_folds)

    print('-' * 25)


    # val_mean,test_mean = train_model(args, datareader, splits)
    test_mean = val_model(args, datareader, splits)
