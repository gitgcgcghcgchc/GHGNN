#单损失函数，batch_size=1
import random
import statistics
import time

import numpy as np
import torch
from torch import nn

from delta_tm.dataset.graph_data_reader_couple import GraphData
from delta_tm.packages.regression_metric import regression_metrics
from delta_tm.packages.save_result import save_id,save_node_emb, save_pool_score


def val_model(args, datareader, splits):

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


    for fold_id in range(args.n_folds):
        if fold_id not in [0]:
            break
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

        model_path0 = "save_result/StructΔTm801/GHG/PoincareBall/2025_10_09_15_47/"  # 这里改为你保存的路径
        model_path = model_path0+f'save_models/{fold_id}.pt'
        model = torch.load(model_path,map_location=args.device)
        # print(model)


        # Total trainable param
        c = 0
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            c += p.numel()
        print('N trainable parameters:', c)

        # validation function
        def val(val_loader, use):
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
                all_protein_couple0=[]
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
                    if data[8][0]=='P06168' and data[9][0]=='P05793':
                        all_protein_couple0.append((data[8][0], data[9][0]))

                        all_node_embs.append(node_emb1.tolist())
                        all_node_embs.append(node_emb2.tolist())

                        pool_score.append(pool_score1.tolist())
                        pool_score.append(pool_score2.tolist())

                    n_samples += data[0].size(0)
            return val_loss/n_samples, all_graph_embs, all_preds_couple,all_trues_couple, all_protein_couple0, all_node_embs,pool_score, total_time_iter/n_samples

        # Loss function
        loss_fn = nn.SmoothL1Loss(beta=1.0)

        test_loss, all_graph_embs_test, all_preds_couple_test,all_trues_couple_test, all_protein_couple_test0,all_node_emb_test,pool_score_test, test_time = val(
            loaders[2], 'test' )

        metrics2_test = regression_metrics(all_preds_couple_test, all_trues_couple_test)

        print(len(pool_score_test),len(all_node_emb_test))
        print('-' * 20)
        print(
                "fold -{} -Test Loss: {:.4f},Test R2: {:.4f}".format(
                    fold_id,
                    test_loss, metrics2_test['R2']))

        best_metrics_test_list.append(metrics2_test)
        # save pre
        if args.save:
            print(len(pool_score_test),len(all_node_emb_test))
            save_node_emb(all_protein_couple_test0,all_node_emb_test, model_path0, 'test', fold_id)
            save_pool_score(all_protein_couple_test0,pool_score_test, model_path0, 'test', fold_id)
            save_id(splits[fold_id], path, fold_id)


    best_r2_test_list = [d['R2'] for d in best_metrics_test_list]

    print('-' * 50)
    print(best_r2_test_list)
    print('{}-fold cross test avg acc (+- std): {} ({})'.format(args.n_folds, statistics.mean(best_r2_test_list),
                                                                statistics.stdev(best_r2_test_list)))

    print('-' * 25)

    return statistics.mean(best_r2_test_list)



