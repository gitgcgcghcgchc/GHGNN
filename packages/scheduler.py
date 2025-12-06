import math
from torch.optim.lr_scheduler import _LRScheduler

class TwoPhaseCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T, T1, T2, base_lr1,base_lr2, eta_min1=0, eta_min2=0, last_epoch=-1):
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.base_lr1 = base_lr1
        self.base_lr2 = base_lr2
        self.eta_min1 = eta_min1
        self.eta_min2 = eta_min2
        super(TwoPhaseCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T:
            # 第1阶段
            T_cur = self.last_epoch
            return [
                self.eta_min1 + (self.base_lr1 - self.eta_min1) *
                (1 + math.cos(math.pi * T_cur / self.T1)) / 2
                for _ in self.optimizer.param_groups
            ]
        elif self.T<self.last_epoch <100 :
            # 第2阶段
            T_cur = self.last_epoch - self.T
            return [
                self.eta_min2 + (self.base_lr2 - self.eta_min2) *
                (1 + math.cos(math.pi * T_cur / self.T2)) / 2
                for _ in self.optimizer.param_groups
            ]
        else:
            # 第2阶段
            T_cur = self.last_epoch - self.T
            return [
                self.eta_min2 + (self.base_lr2/50 - self.eta_min2) *
                (1 + math.cos(math.pi * T_cur / self.T2)) / 2
                for _ in self.optimizer.param_groups
            ]

class WarmupTwoPhaseCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_warmup, T, T1, T2, base_lr1, base_lr2, eta_min1=0, eta_min2=0, last_epoch=-1):
        self.T_warmup = T_warmup  # warm up steps
        self.T = T                # 分界点
        self.T1 = T1              # 第一阶段长度
        self.T2 = T2              # 第二阶段长度
        self.base_lr1 = base_lr1
        self.base_lr2 = base_lr2
        self.eta_min1 = eta_min1
        self.eta_min2 = eta_min2
        super(WarmupTwoPhaseCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Phase 0: Warm up
        if step < self.T_warmup:
            warmup_lr = self.base_lr1 * step / float(self.T_warmup)+0.005
            return [warmup_lr for _ in self.optimizer.param_groups]

        # Phase 1
        elif self.T_warmup <= step < self.T:
            T_cur = step - self.T_warmup
            return [
                self.eta_min1 + (self.base_lr1 - self.eta_min1) *
                (1 + math.cos(math.pi * T_cur / self.T1)) / 2
                for _ in self.optimizer.param_groups
            ]

        # Phase 2
        elif self.T <= step < 100:
            T_cur = step - self.T
            return [
                self.eta_min2 + (self.base_lr2 - self.eta_min2) *
                (1 + math.cos(math.pi * T_cur / self.T2)) / 2
                for _ in self.optimizer.param_groups
            ]

        # Phase 3: late decay
        else:
            T_cur = step - self.T
            return [
                self.eta_min2 + (self.base_lr2 / 50 - self.eta_min2) *
                (1 + math.cos(math.pi * T_cur / self.T2)) / 2
                for _ in self.optimizer.param_groups
            ]