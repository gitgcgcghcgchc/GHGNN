import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

def plot_label_distributions(labels, fold_id):
    """
    绘制训练集、验证集、测试集标签分布
    labels: 一个包含训练集、验证集、测试集标签的列表
    fold_id: 当前的折号，用于保存不同折的图像
    """
    plt.figure(figsize=(8, 5))

    # 分别绘制训练集、验证集、测试集的标签分布
    splits = ['Train', 'Validation', 'Test']
    colors = ['blue', 'orange', 'green']

    for i, (split_labels, split_name, color) in enumerate(zip(labels, splits, colors)):
        # 使用 Yeo-Johnson 变换（适用于正负数据）
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_labels = transformer.fit_transform(np.array(split_labels).reshape(-1, 1)).flatten()
        sns.histplot(transformed_labels, bins=30, kde=True, label=split_name, color=color, element='step',
                     fill=True)

    plt.title(f'Label Distributions - Fold {fold_id}')
    plt.xlabel('Label Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'label_distributions_fold{fold_id}.png')
    # plt.close()


