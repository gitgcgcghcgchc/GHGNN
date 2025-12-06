#单损失函数，batch_size=1
import importlib
import os
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler

from delta_tm.dataset.graph_data_reader_couple import GraphData
from delta_tm.packages.add_noise import add_gradient_noise
from delta_tm.packages.check_grad import check_layerwise_gradients

from delta_tm.packages.path import make_path
from delta_tm.packages.plot import make_plot
from delta_tm.packages.regression_metric import regression_metrics
from delta_tm.packages.reset_model import reset_model_weights, reset_model
from delta_tm.packages.save_result import save_plot_figure, save_pre, save_emb, save_id, save_args, save_result, \
    save_node_emb, save_pool_score
from torch_optimizer import Lookahead
from dadaptation import DAdaptAdan
import graphzoo as gz
from delta_tm.packages.scheduler import TwoPhaseCosineAnnealingLR, WarmupTwoPhaseCosineAnnealingLR


def train_model(args, datareader, splits):

    def seed_worker(seed):
        # 为当前工作进程设置种子
        worker_seed = torch.initial_seed() % 2 ** 32  # 确保种子在32位范围内
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    time_start = time.time()
    # Train & test each fold
    best_metrics_val_list, best_metrics_test_list = [], []

    time_folds = []
    path = 'save_result/' + f'{args.dataset}/{args.model}/' + args.manifold

    if args.save ==1:
        path = make_path(path)
        save_args(args, path + '/args.txt')
    for fold_id in range(args.n_folds):
        # if fold_id in [0]:
        #     continue
        # print('-' * 25)
        print('Fold:', fold_id)
        loss_train_fold, loss_val_fold, loss_test_fold = [], [], []
        r2_train_fold, r2_val_fold, r2_test_fold = [], [], []

        loaders = []
        deltatm =[]
        for split in ['train', 'val', 'test']:
            # Build GDATA object
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,
                              split=split,
                              splits=splits)
            deltatm.append(gdata.deltatm)
            # Build graph data pytorch loader
            loader = torch.utils.data.DataLoader(gdata,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,  # 是否在每个 epoch 开始时随机打乱数据
                                                 num_workers=4,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 worker_init_fn=seed_worker)

            loaders.append(loader)

        model = getattr(importlib.import_module('models.' + f'{args.model}'), args.model)(args).to(args.device)
        print(model)
        # Optimizer
        # base_optimizer =torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        base_optimizer = gz.optimizers.RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(base_optimizer)

        # Total trainable param
        c = 0
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            c += p.numel()
        print('N trainable parameters:', c)
        # plot_label_distributions(deltatm, fold_id)
        scheduler_cosine = TwoPhaseCosineAnnealingLR(optimizer, T=50, T1=50, T2=30, base_lr1=args.lr,
                                                     base_lr2=args.lr / 50, eta_min1=1e-6,
                                                     eta_min2=1e-6)
        scheduler_cosine1 = TwoPhaseCosineAnnealingLR(optimizer, T=10, T1=10, T2=30, base_lr1=args.lr,
                                                     base_lr2=args.lr / 50, eta_min1=1e-6, eta_min2=1e-6)
        scaler = GradScaler()

        # Train function
        def train(train_loader, epoch, step):

            print('-' * 30)
            print('train model ...')
            total_time_iter = 0
            model.train()
            start = time.time()
            train_loss, n_samples, correct = 0, 0, 0


            all_preds, all_preds_couple = [], []
            all_trues, all_trues_couple = [], []
            all_protein_couple = []
            all_graph_embs = []

            for batch_idx, data in enumerate(train_loader):
                for i in range(len(data)):
                    if isinstance(data[i], list):
                        continue
                    if args.cuda==0:
                        data[i] = data[i].to(args.device)

                # output, graph_emb = model(data)
                output, graph_emb, _, _,_,_ = model(data)
                loss = loss_fn(output[2], data[-2])

                # 计算损失并反向传播
                scaler.scale(loss).backward()

                if epoch <50:
                    # 添加噪声
                    global_step = epoch * len(train_loader) + batch_idx
                    add_gradient_noise(model.parameters(), eta=0.1, gamma=0.55, step=global_step)

                if (batch_idx + 1) % args.loss_item == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # 清零梯度
                    # optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()

                time_iter = time.time() - start
                total_time_iter += time_iter

                #
                all_preds.extend(output[0].detach().cpu().numpy())
                all_preds.extend(output[1].detach().cpu().numpy())
                all_preds_couple.extend(output[2].detach().cpu().numpy())

                all_trues.extend(data[6].detach().cpu().numpy())
                all_trues.extend(data[7].detach().cpu().numpy())
                all_trues_couple.extend((data[-2]).detach().cpu().numpy())

                all_graph_embs.append(graph_emb[0].tolist())
                all_graph_embs.append(graph_emb[1].tolist())

                all_protein_couple.append((data[8], data[9]))
                del output, graph_emb
                n_samples += data[0].size(0)

            # 计算指标
            # metrics2 = regression_metrics(all_preds_couple, all_trues_couple)

            return train_loss/n_samples, all_graph_embs, all_preds, all_preds_couple, all_trues, all_trues_couple,  all_protein_couple, total_time_iter/n_samples

        # validation function
        def val(val_loader, use, step):
            total_time_iter = 0
            start = time.time()
            print(f'{use} model ...')
            model.eval()
            with torch.no_grad():
                start = time.time()
                val_loss,  correct, n_samples = 0, 0, 0
                all_preds, all_preds_couple = [], []
                all_trues, all_trues_couple = [], []
                all_protein_couple = []
                all_graph_embs = []
                all_node_embs = []
                pool_score=[]


                for batch_idx, data in enumerate(val_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], list):
                            continue
                        if args.cuda==0:
                            data[i] = data[i].float().to(args.device)

                    # output, graph_emb = model(data)
                    output, graph_emb, node_emb1, node_emb2,pool_score1, pool_score2 = model(data)

                    loss = loss_fn(output[2], data[-2])
                    val_loss += loss.item()

                    time_iter = time.time() - start
                    total_time_iter += time_iter

                    # 收集预测值和真实标签，用于计算精确率、召回率和F1分数
                    all_preds.extend(output[0].detach().cpu().numpy())
                    all_preds.extend(output[1].detach().cpu().numpy())
                    all_preds_couple.extend(output[2].detach().cpu().numpy())

                    all_trues.extend(data[6].detach().cpu().numpy())
                    all_trues.extend(data[7].detach().cpu().numpy())
                    all_trues_couple.extend((data[-2]).detach().cpu().numpy())

                    all_graph_embs.append(graph_emb[0].tolist())
                    all_graph_embs.append(graph_emb[1].tolist())

                    all_protein_couple.append((data[8][0], data[9][0]))
                    if data[8][0]=='P56690' and data[9][0]== 'P41252':
                        all_node_embs.append(node_emb1.tolist())
                        all_node_embs.append(node_emb2.tolist())

                        pool_score.append(pool_score1.tolist())
                        pool_score.append(pool_score2.tolist())

                    n_samples += data[0].size(0)
            # time_iter = time.time() - start

            # 计算指标
            # metrics1 = regression_metrics(all_preds, all_trues)
            # metrics2 = regression_metrics(all_preds_couple, all_trues_couple)

            return val_loss/n_samples, all_graph_embs, all_preds, all_preds_couple, all_trues, all_trues_couple, all_protein_couple, all_node_embs,pool_score, total_time_iter/n_samples

        # Loss function
        loss_fn = nn.SmoothL1Loss(beta=1.0)

        total_time = 0
        best_epoch = 0
        best_r2_train_couple, best_r2_val_couple, best_r2_test_couple = -1, -1, -1

        best_metrics_val, best_metrics_test = {}, {}

        patience = 0
        train_step = 0

        step_1_epoch = 0


        for epoch in range(args.epochs):

            dead_relu_stats = {}
            def relu_hook(name):
                def hook(module, input, output):
                    zero_ratio = (output == 0).float().mean().item()
                    dead_relu_stats[name] = zero_ratio

                return hook

            # 注册 hook 到所有 ReLU 层
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, nn.ReLU):
                    hooks.append(module.register_forward_hook(relu_hook(name)))


            if train_step == 0:
                for param in model.parameters():
                    param.requires_grad = True

            elif train_step == 1:
            # 冻结 layers 部分的参数
                for param in model.parameters():
                    param.requires_grad = False
            for param in model.mlp_couple1.parameters():
                param.requires_grad = True


            train_loss, all_graph_embs_train, all_preds_train, all_preds_couple_train, all_trues_train, all_trues_couple_train, all_protein_couple_train, train_time = train(
                loaders[0], epoch, train_step)
            val_loss, all_graph_embs_val, all_preds_val, all_preds_couple_val, all_trues_val, all_trues_couple_val, all_protein_couple_val,_,_,val_time = val(
                loaders[1], 'val', train_step)
            test_loss, all_graph_embs_test, all_preds_test, all_preds_couple_test, all_trues_test, all_trues_couple_test, all_protein_couple_test,all_node_emb_test,pool_score_test, test_time = val(
                loaders[2], 'test', train_step )

            if np.isnan(all_preds_couple_train).any() or np.isnan(all_preds_couple_val).any() or np.isnan(all_preds_couple_test).any():
                print("出现NAN，自动重新训练...")
                model, optimizer, scheduler_cosine, scheduler_cosine1, scaler = reset_model(args)
                continue

            metrics2_train = regression_metrics(all_preds_couple_train, all_trues_couple_train)
            metrics2_val = regression_metrics(all_preds_couple_val, all_trues_couple_val)
            metrics2_test = regression_metrics(all_preds_couple_test, all_trues_couple_test)

            # reset = check_layerwise_gradients(model)
            if epoch>30 and metrics2_val['R2']<0.:
                reset = 1
            else:
                reset = 0
            if reset == 1:
                print("梯度消失，自动重新训练...")
                reset_model_weights(model)
                base_optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # base_optimizer = gz.optimizers.RiemannianAdam(model.parameters(), args.lr, betas=[0.9, 0.95],
                #                                               weight_decay=args.weight_decay, eps=1e-8, )  # 优化器
                # base_optimizer = DAdaptAdan(model.parameters(), weight_decay=args.weight_decay)
                optimizer = Lookahead(base_optimizer)
                scheduler_cosine = TwoPhaseCosineAnnealingLR(optimizer, T=50, T1=50, T2=30, base_lr1=args.lr,
                                                             base_lr2=args.lr / 50, eta_min1=1e-6,
                                                             eta_min2=1e-6)
                scheduler_cosine1 = TwoPhaseCosineAnnealingLR(optimizer, T=10, T1=10, T2=30, base_lr1=args.lr,
                                                             base_lr2=args.lr / 50, eta_min1=1e-6,
                                                             eta_min2=1e-6)


            print('-' * 20)
            print(
                "fold -{} -Epoch - {:03d}({:03d}) - Step {:01d}\n Train Loss: {:.4f}, Train R2: {:.4f}({:.4f})\n Validation Loss {:.4f}, Validation R2: {:.4f}({:.4f})\n Test Loss: {:.4f},Test R2: {:.4f}({:.4f})".format(
                    fold_id, epoch + 1, best_epoch+1, train_step, train_loss, metrics2_train['R2'], best_r2_train_couple,
                    val_loss, metrics2_val['R2'], best_r2_val_couple,
                    test_loss, metrics2_test['R2'], best_r2_test_couple, ))
            #设置学习率变化器
            if train_step == 0:
                scheduler_cosine.step()
            else:
                scheduler_cosine1.step()


            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"- Current Learning Rate: {current_lr}")
            print('patience is:', patience)

            loss_train_fold.append(round(train_loss, 2))
            loss_val_fold.append(round(val_loss, 2))
            loss_test_fold.append(round(test_loss, 2))

            r2_train_fold.append(round(metrics2_train['R2'], 2))
            r2_val_fold.append(round(metrics2_val['R2'], 2))
            r2_test_fold.append(round(metrics2_test['R2'], 2))

            # 第一阶段：至少运行 args.min_epochs[0] 次
            if train_step == 0:
                if epoch < args.min_epochs[0]:
                    pass  # 继续第一阶段
                else:
                    if metrics2_val['R2'] > best_r2_val_couple:
                        patience = 0
                    else:
                        patience += 1
                    if patience > args.patience:
                        patience = 0
                        train_step = 1  # 进入第二阶段
                        step_1_epoch = epoch
                        model = best_model

            # 第二阶段：至少运行 args.min_epochs[1] 次
            elif train_step == 1:
                if epoch < step_1_epoch + args.min_epochs[1]:
                    pass  # 继续第三阶段
                else:
                    if metrics2_val['R2'] > best_r2_val_couple:
                        patience = 0
                    else:
                        patience += 1
                    if patience > args.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break


            if metrics2_val['R2'] > best_r2_val_couple:
                best_epoch = epoch

                best_metrics_val = metrics2_val
                best_metrics_test = metrics2_test

                best_r2_train_couple = metrics2_train['R2']
                best_r2_val_couple = metrics2_val['R2']
                best_r2_test_couple = metrics2_test['R2']

                protein_couple_train_fold, all_trues_train_fold, all_pres_train_fold, all_trues_couple_train_fold, all_preds_couple_train_fold, all_graph_embs_train_fold = all_protein_couple_train, all_trues_train, all_preds_train, all_trues_couple_train, all_preds_couple_train, all_graph_embs_train
                protein_couple_val_fold, all_trues_val_fold, all_pres_val_fold, all_trues_couple_val_fold, all_preds_couple_val_fold, all_graph_embs_val_fold = all_protein_couple_val, all_trues_val, all_preds_val, all_trues_couple_val, all_preds_couple_val, all_graph_embs_val
                protein_couple_test_fold, all_trues_test_fold, all_pres_test_fold, all_trues_couple_test_fold, all_preds_couple_test_fold, all_graph_embs_test_fold = all_protein_couple_test, all_trues_test, all_preds_test, all_trues_couple_test, all_preds_couple_test, all_graph_embs_test
                all_node_emb_test_fold = all_graph_embs_test
                pool_score_test_fold = pool_score_test
                best_model = model

        # if fold_id == 0 and best_r2_test_couple < 0.70:
        #     print(f"Early stop: Fold 0 score {best_r2_test_couple:.4f} < 0.72")
        #     return -1.0, -1.0  # 用无效值代表被剪枝


        best_metrics_val_list.append(best_metrics_val)
        best_metrics_test_list.append(best_metrics_test)

        time_folds.append(round(total_time / args.epochs, 2))

        make_plot(loss_train_fold, loss_val_fold, loss_test_fold, r2_train_fold, r2_val_fold, r2_test_fold)

        save_plot_figure(fold_id, path)

        # save pre
        if args.save:
            save_pre(protein_couple_train_fold, all_pres_train_fold, all_trues_train_fold, all_preds_couple_train_fold,
                     all_trues_couple_train_fold, path, 'train', fold_id)
            save_pre(protein_couple_val_fold, all_pres_val_fold, all_trues_val_fold, all_preds_couple_val_fold,
                     all_trues_couple_val_fold, path, 'val', fold_id)
            save_pre(protein_couple_test_fold, all_pres_test_fold, all_trues_test_fold, all_preds_couple_test_fold,
                     all_trues_couple_test_fold, path, 'test', fold_id)

            save_emb(protein_couple_train_fold, all_graph_embs_train_fold, path, 'train', fold_id)
            save_emb(protein_couple_val_fold, all_graph_embs_val_fold, path, 'val', fold_id)
            save_emb(protein_couple_test_fold, all_graph_embs_test_fold, path, 'test', fold_id)

            save_id(splits[fold_id], path, fold_id)
        # Save model
        if args.save_model:
            # print('Save model ...')
            file_path = path + '/save_models'
            # 创建目录
            os.makedirs(file_path, exist_ok=True)
            torch.save(best_model, file_path + '/' + str(fold_id) + f'.pt')
            # print('Complete to save model')

    best_r2_val_list = [d['R2'] for d in best_metrics_val_list]
    best_r2_test_list = [d['R2'] for d in best_metrics_test_list]

    print('-' * 50)
    print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds,
                                                                      statistics.mean(best_r2_val_list),
                                                                      statistics.stdev(best_r2_val_list)))

    print(best_r2_test_list)
    print('{}-fold cross test avg acc (+- std): {} ({})'.format(args.n_folds, statistics.mean(best_r2_test_list),
                                                                statistics.stdev(best_r2_test_list)))

    time_end = time.time()
    time_total = time_end - time_start
    save_result(path, args, best_metrics_val_list, best_metrics_test_list, time_total, train_time, val_time, test_time)

    print('-' * 25)

    return statistics.mean(best_r2_val_list), statistics.mean(best_r2_test_list)



