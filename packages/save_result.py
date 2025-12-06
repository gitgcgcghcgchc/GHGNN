import csv
import itertools
import os
import statistics

import numpy as np
from matplotlib import pyplot as plt


def create_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
    except OSError:
        print('Creating', directory_path, 'directory error')

def save_result_csv(file_path, result, print_column_name=False, mode='a'):
    with open(file_path, mode, newline='\n') as file:
        writer = csv.writer(file)
        if print_column_name:
            writer.writerow(['', 'mean', '(std)'])
        writer.writerow(result)

def save_plot_figure(fold, new_folder_name):
    fig = plt.gcf()
    file_path = new_folder_name+'/figure'
    plot_filename = f"fold{fold}.png"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    plot_filepath = os.path.join(file_path, plot_filename)

    fig.savefig(plot_filepath)
    plt.close()

def save_args(args, file_path):
    # 确保父目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 写入文件
    with open(file_path, "w") as f:
        for name, value in vars(args).items():
            f.write(f"{name}: {value}\n")  # 每行格式：参数名: 值

def list_to_dict(list_of_dicts):
    # 提取所有键
    keys = list(list_of_dicts[0].keys())

    # 初始化结果字典
    result = {key: [] for key in keys}

    # 提取每个键的值
    for d in list_of_dicts:
        for key in keys:
            result[key].append(d[key])

    # 计算每个列表的平均值和标准差
    for key in keys:
        values = result[key]
        mean_value = np.mean(values)
        std_value = np.std(values)
        result[key].append(mean_value)
        result[key].append(std_value)

    return result

def save_result(path, args, best_val_list,best_test_list, total_time, train_time, val_time=0, test_time=0):
    # Save 10-cross validation result as csv format

    best_val_list = list_to_dict(best_val_list)
    best_test_list = list_to_dict(best_test_list)

    path1 = path + '/result.csv'

    for key in best_val_list.keys():
        save_result_csv(path1, [f'{key}'])
        save_result_csv(path1, best_val_list[key])
        save_result_csv(path1, best_test_list[key])

    save_result_csv(path1,['time'])
    save_result_csv(path1, [total_time, train_time, val_time, test_time])


def save_id(id,file,fold):
    file_path = file + '/' + 'id'
    create_directory(file_path)
    file_name = file_path + '/' + str(fold) + '.csv'
    with open(file_name, "w", encoding="utf-8") as file:
        for key, values in id.items():
            # 将键和值拼接成一行，用空格分隔
            line = f"{key} {' '.join(map(str, values))}\n"
            file.write(line)

def save_pre(protein_couple, pre,true, pre_couple, true_couple, file, use, fold):

    pre = list(zip(pre[::2], pre[1::2]))
    true = list(zip(true[::2], true[1::2]))

    file_path = file+'/'+'pre/'+use
    create_directory(file_path)
    file_name = file_path+'/'+str(fold)+'.csv'
    with open(file_name, "w", encoding="utf-8") as file:
        # 写入列名
        file.write(f"protein1\tprotein2\tpre1\tpre2\ttrue1\ttrue2\tpre_couple\ttrue_couple\n")
        # 写入数据

        for (item1, item2), (item3,item4), (item5,item6), item7, item8 in zip(protein_couple,pre,true,pre_couple,true_couple):
            file.write(f"{item1}\t{item2}\t{item3}\t{item4}\t{item5}\t{item6}\t{item7}\t{item8}\n")

def save_emb(protein_couple,graph_embs, file, use, fold):

    protein_couple = [item for pair in protein_couple for item in pair]

    file_path = file + '/' + 'emb/' + use
    create_directory(file_path)
    file_name = file_path + '/' + str(fold) + '.csv'
    with open(file_name, "w", encoding="utf-8") as file:
        # 写入列名
        file.write(f"protein\temb\n")
        for protein, emb in zip(protein_couple,graph_embs):
            # 将子列表中的元素转换为字符串，并用空格隔开
            line = " ".join(map(str, emb))
            file.write(f"{protein},{line}\n")  # 写入文件，每行一个子列表

def save_node_emb(protein_couple, node_embs, file, use, fold):
    protein_couple0 = list(itertools.chain(*protein_couple))
    file_path0 = file + '/' + 'nod_emb/' + use + '/'+str(fold)
    create_directory(file_path0)

    for matrix, name in zip(node_embs, protein_couple0):
        # 构建文件路径
        print(name)
        file_path = os.path.join(file_path0, f"{name}.npy")
        if name =='P06168' or name =='P05793':
            print(name)
            # 保存矩阵到文件
            np.save(file_path, matrix)

def save_pool_score(protein_couple, pool_score, file, use, fold):
    protein_couple0 = list(itertools.chain(*protein_couple))
    file_path0 = file + '/' + 'pool_score/' + use + '/' + str(fold)
    create_directory(file_path0)

    for matrix, name in zip(pool_score, protein_couple0):
        # 构建文件路径
        file_path = os.path.join(file_path0, f"{name}.npy")
        if name == 'P06168' or name =='P05793':
            print(name)
            # 保存矩阵到文件
            np.save(file_path, matrix)
