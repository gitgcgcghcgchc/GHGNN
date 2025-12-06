import importlib

import torch
from torch.cuda.amp import GradScaler
from torch_optimizer import Lookahead

from delta_tm.packages.scheduler import TwoPhaseCosineAnnealingLR


def reset_model_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        # 如果是嵌套结构（如Sequential或ModuleList），递归调用
        elif isinstance(layer, torch.nn.Module):
            reset_model_weights(layer)

def reset_model(args):

    model = getattr(importlib.import_module('models.' + f'{args.model}'), args.model)(args).to(args.device)
    base_optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Lookahead(base_optimizer)
    scheduler_cosine = TwoPhaseCosineAnnealingLR(optimizer, T=50, T1=50, T2=30, base_lr1=args.lr,
                                                 base_lr2=args.lr / 50, eta_min1=1e-6,
                                                 eta_min2=1e-6)
    scheduler_cosine1 = TwoPhaseCosineAnnealingLR(optimizer, T=10, T1=10, T2=30, base_lr1=args.lr,
                                                  base_lr2=args.lr / 50, eta_min1=1e-6, eta_min2=1e-6)
    scaler = GradScaler()
    return model, optimizer, scheduler_cosine,scheduler_cosine1, scaler