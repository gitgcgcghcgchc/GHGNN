import os
import warnings
from scipy import stats
import torch

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from delta_tm.dataset.packages import save_node_features, save_species, get_specieses, get_matrix, get_species, \
    get_network, make_seq, save_adj, get_aaindex_feature, floyd_warshall, pca_reduce_aaindex
from sklearn.preprocessing import PowerTransformer
from delta_tm.position_embeddings.absolute_position import get_absolute_positional_encoding

def make_data(dataset):
    # 指定文件路径
    file_path = f'dataset/{dataset}/real_couple.xlsx'  # 替换为你的文件路径

    # 读取Excel文件，指定第2行为列名（header=1），并只读取前5列（usecols=[0,1,3,4,5]，跳过第3列）
    df = pd.read_excel(file_path, header=1, usecols=[0, 1, 3, 4],nrows=801, engine='openpyxl')

    specieses = get_species(f"dataset/{dataset}/PDB")
    save_species(specieses, f'dataset/{dataset}/species.txt')
    specieses = get_specieses(f'dataset/{dataset}/species.txt')
    # 将字典添加到DataFrame的新列中

    name, networks, networks_adj, seq = get_network(f"dataset/{dataset}/wl")
    save_adj(networks_adj,f"dataset/{dataset}/adj")
    seq = make_seq(seq)
    save_node_features(seq,f"dataset/{dataset}/feature")

    short_dists = floyd_warshall(networks_adj)

    save_adj(short_dists,f"dataset/{dataset}/short_dist")

    networks_adj = get_matrix(f'dataset/{dataset}/adj')
    features =get_matrix(f'dataset/{dataset}/feature')
    dists =get_matrix(f'dataset/{dataset}/short_dist')

    df['TM1']=df['TM1'].astype('float32')
    df['TM2']=df['TM2'].astype('float32')

    df['species1'] = df['protein1'].map(specieses)
    df['species2'] = df['protein2'].map(specieses)

    df['adj1'] = df['protein1'].map(networks_adj)
    df['adj2'] = df['protein2'].map(networks_adj)

    df['feature1'] = df['protein1'].map(features)
    df['feature2'] = df['protein2'].map(features)

    df['seq1'] = df['protein1'].map(seq)
    df['seq2'] = df['protein2'].map(seq)



    # 查看数据
    print("数据已成功加载到DataFrame！")
    # 保存为 HDF5 文件
    df.to_hdf(f'dataset/{dataset}/dataset.h5', key='df', mode='w')
    print(f"DataFrame 已保存")

def get_data(dataset = 'StructΔTm801', rnd_state=17, folds=10, root_file='dataset/'):
    rnd_state = np.random.RandomState(rnd_state)
    file_path = f'{dataset}/dataset.h5'
    print('dataset:', dataset)

    file_path = root_file+file_path
    print(file_path)
    # 判断文件是否存在
    if os.path.exists(file_path):
        print(f"文件 {file_path} 已存在，已加载数据。")
    else:
        make_data(dataset)
        # 文件存在，读取 HDF5 文件
    df = pd.read_hdf(file_path, key='df')

    aaindex_pca = pca_reduce_aaindex(root_file+'aaindex_feature.txt',20)

    df['aaindex_feature1'] = df['seq1'].apply(lambda x: get_aaindex_feature(x, aaindex_pca))
    df['aaindex_feature2'] = df['seq2'].apply(lambda x: get_aaindex_feature(x, aaindex_pca))

    dists = get_matrix(root_file+f'{dataset}/short_dist')
    df['short_dist1'] = df['protein1'].map(dists)
    df['short_dist2'] = df['protein2'].map(dists)

    k = df['feature1'][0].shape[1]+df['aaindex_feature1'][0].shape[1]
    print('数据原始特征维度为：', k)
    # 生成 'generated' 列
    # df['position_emb1'] = df['adj1'].apply(lambda x: get_absolute_positional_encoding(x.shape[0], k))
    # df['position_emb2'] = df['adj2'].apply(lambda x: get_absolute_positional_encoding(x.shape[0], k))


    # 创建 Yeo-Johnson 变换器
    transformer = PowerTransformer(method='yeo-johnson')

    # 对 df['TM'] 列进行 Yeo-Johnson 变换
    # 需要将 'TM' 列转换为二维数组格式进行变换
    df['delta_TM'] = df['TM1']-df['TM2']
    df['delta_TM_transformed'] = transformer.fit_transform((df['TM1']-df['TM2']).values.reshape(-1, 1))*10



    # # Shapiro-Wilk 正态性检验
    # stat, p_value = stats.shapiro(df['delta_TM_transformed'])
    # print(f"Shapiro-Wilk Test Statistic: {stat}")
    # print(f"Shapiro-Wilk Test p-value: {p_value}")
    #
    # # 根据 p-value 判断是否服从正态分布
    # if p_value > 0.05:
    #     print("数据服从正态分布")
    # else:
    #     print("数据不服从正态分布")

    # 获取行数和列数
    N_couples, cols = df.shape

    print('-'*50)
    # 获取 DataFrame 的内存使用情况
    df.info(memory_usage='deep')
    # Create test sets first
    remaining_ids, train_ids, val_ids, test_ids = split_ids(np.arange(N_couples), rnd_state=rnd_state, folds=folds)

    # Create train sets
    splits = []
    for fold in range(folds):
        splits.append({'main':remaining_ids,
                        'train': train_ids[fold],
                       'val': val_ids[fold],
                       'test': test_ids})
    return df, splits

def split_ids(ids_all, rnd_state=None, folds=10, test_size=0.2):
    n = len(ids_all)

    # 第一步：划分出 10% 的数据作为测试集
    test_size = int(n * test_size)  # 测试集的大小
    ids = ids_all[rnd_state.permutation(n)]  # 打乱数据
    test_ids = ids[:test_size]  # 选择前 10% 作为测试集
    remaining_ids = ids[test_size:]  # 剩余的 90% 数据

    # 第二步：将剩余的 90% 数据分成 folds 份作为训练集和验证集
    stride = int(np.ceil(len(remaining_ids) / float(folds)))  # 计算每一折的大小
    validation_ids = [remaining_ids[i: i + stride] for i in range(0, len(remaining_ids), stride)]  # 划分验证集

    # 确保每个数据集没有丢失
    assert np.all(np.unique(np.concatenate(validation_ids)) == sorted(
        remaining_ids)), 'some graphs are missing in the validation sets'
    assert len(validation_ids) == folds, 'invalid validation sets'

    train_ids = []
    for fold in range(folds):
        # 训练集是去掉当前验证集的数据
        train_ids.append(np.array([e for e in remaining_ids if e not in validation_ids[fold]]))

    # for fold in range(folds):
    #     # 计算需要移动的 ID 数量
    #     move_size = int(len(train_ids[fold]) * 0.05)
    #     if move_size == 0:
    #         continue  # 避免空操作
    #
    #     # 随机抽取 ID
    #     moved_ids = rnd_state.choice(train_ids[fold], size=move_size, replace=False)
    #     removed_ids = rnd_state.choice(test_ids, size=move_size, replace=False)
    #     test_ids = np.array([e for e in test_ids if e not in removed_ids])
    #     # 将抽出的 ID 加入测试集
    #     test_ids = np.concatenate([test_ids, moved_ids])

        # 确保每个训练集和验证集的组合没有错误
        assert len(train_ids[fold]) + len(validation_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(validation_ids[fold]))) == len(remaining_ids), 'invalid splits'

    return remaining_ids, train_ids, validation_ids, test_ids


class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 fold_id,
                 datareader,
                 split,
                 splits,
                 ):
        self.short_dist2 = None
        self.short_dist1 = None
        self.aaindex_feature2 = None
        self.aaindex_feature1 = None
        self.protein2 = None
        self.protein1 = None
        self.indices = None
        self.features2 = None
        self.features1 = None
        self.adj2 = None
        self.adj1 = None
        self.species1 = None
        self.species2 = None
        self.idx = None
        self.features_dim = None
        self.tm1 = None
        self.tm2 = None
        self.position_emb1 = None
        self.position_emb2 = None
        self.delta_tm = None
        self.set_fold(datareader, splits, split, fold_id)
        self.N_nodes_max = 2214

    def set_fold(self, data, splits, split, fold_id):
        # Initialize common attributes
        self.features_dim = data['feature1'][1].shape[1]
        self.idx = splits[fold_id][split]
        idx_subset = self.idx


        print(f'number of {split} couple:',len(self.idx))

        self.tm1 = [float(data['TM1'][i]) for i in idx_subset]
        self.tm2 = [float(data['TM2'][i]) for i in idx_subset]

        self.delta_tm = [float(data['delta_TM_transformed'][i]) for i in idx_subset]
        self.deltatm = [a - b for a, b in zip(self.tm1, self.tm2)]

        self.species1 = [data['species1'][i] for i in idx_subset]
        self.species2 = [data['species2'][i] for i in idx_subset]

        self.adj1 = [data['adj1'][i] for i in idx_subset]
        self.adj2 = [data['adj2'][i] for i in idx_subset]

        self.features1 = [data['feature1'][i] for i in idx_subset]
        self.features2 = [data['feature2'][i] for i in idx_subset]

        self.protein1 = [data['protein1'][i] for i in idx_subset]
        self.protein2 = [data['protein2'][i] for i in idx_subset]

        self.short_dist1 = [data['short_dist1'][i] for i in idx_subset]
        self.short_dist2 = [data['short_dist2'][i] for i in idx_subset]

        self.aaindex_feature1 = [data['aaindex_feature1'][i] for i in idx_subset]
        self.aaindex_feature2 = [data['aaindex_feature2'][i] for i in idx_subset]

        # self.position_emb1 = [data['position_emb1'][i] for i in idx_subset]
        # self.position_emb2 = [data['position_emb2'][i] for i in idx_subset]
        # Adjust indices based on subsampling
        self.indices = np.arange(len(idx_subset))


    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):
        sz = mtx.shape
        # assert len(sz) == 2, ('only 2d arrays are supported', sz)

        if len(sz) == 2:
            if desired_dim2 is not None:
                mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant',
                             constant_values=value)
            else:
                mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        elif len(sz) == 3:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0), (0, 0)), 'constant', constant_values=value)

        return mtx

    def nested_list_to_torch(self, data):
        for i in range(len(data)):
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            elif isinstance(data[i], list):
                data[i] = torch.tensor(data[i], dtype=torch.float32)
            elif isinstance(data[i], str):
                pass

        return data

    def __len__(self):
        return len(self.tm1)

    def __getitem__(self, index):
        index = self.indices[index]

        self.features1[index].data = self.features1[index].data.astype('float32')
        self.features2[index].data = self.features2[index].data.astype('float32')

        self.adj1[index].data = self.adj1[index].data.astype('float32')
        self.adj2[index].data = self.adj2[index].data.astype('float32')

        self.short_dist1[index].data = self.short_dist1[index].data.astype('float32')
        self.short_dist2[index].data = self.short_dist2[index].data.astype('float32')


        # Use nested_list_to_torch with the necessary parameters
        # torch = self.nested_list_to_torch([
        #     self.pad(self.features1[index].todense(), self.N_nodes_max),  # Node features (no need for copy if not modified)
        #     self.pad(self.features2[index].todense(), self.N_nodes_max),
        #
        #     self.pad(self.adj1[index].todense(), self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
        #     self.pad(self.adj2[index].todense(), self.N_nodes_max, self.N_nodes_max),
        #
        #     self.species1[index],
        #     self.species2[index],
        #
        #     self.tm1[index],
        #     self.tm2[index],
        #
        #     self.protein1[index],
        #     self.protein2[index],
        #
        #     self.deltatm[index],
        #     self.delta_tm[index],
        # ])
        torch = self.nested_list_to_torch([
            self.features1[index].todense(),
            self.features2[index].todense(),

            self.adj1[index].todense(),  # Adjacency matrix
            self.adj2[index].todense(),

            self.species1[index],
            self.species2[index],

            self.tm1[index],
            self.tm2[index],

            self.protein1[index],
            self.protein2[index],

            self.short_dist1[index].todense(),
            self.short_dist2[index].todense(),

            self.aaindex_feature1[index],
            self.aaindex_feature2[index],

            # self.position_emb1[index],
            # self.position_emb2[index],

            self.deltatm[index],
            self.delta_tm[index],
        ])
        return torch

if __name__ == '__main__':
    get_data()
