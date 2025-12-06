import numpy as np
from matplotlib import pyplot as plt


def make_plot(loss_train_fold,loss_test_fold,r2_train_fold,r2_test_fold,loss_val_fold=None,r2_val_fold=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Loss Curves')
    plt.plot(range(len(loss_train_fold)), loss_train_fold, label='Training Loss')
    if loss_val_fold != None:
        plt.plot(range(len(loss_val_fold)), loss_val_fold, label='Validation Loss', color='orange')
    plt.plot(range(len(loss_test_fold)), loss_test_fold, label='Test Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('R2 Curves')
    plt.plot(range(len(r2_train_fold)), r2_train_fold, label='Train R2')
    if r2_val_fold != None:
        plt.plot(range(len(r2_val_fold)), r2_val_fold, label='Validation R2', color='orange')
    plt.plot(range(len(r2_test_fold)), r2_test_fold, label='Test R2', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    # 设置 y 轴的网格间隔为 0.1
    y_ticks = list(np.arange(0., 1., 0.1))  # 从 -0.2 到 1.0，间隔为 0.1
    plt.yticks(y_ticks)

    # 找到每个数据集的最大值及其对应的 x 轴位置
    max_train_r2 = max(r2_train_fold)
    max_train_r2_index = r2_train_fold.index(max_train_r2)

    max_train_r2_test = r2_test_fold[max_train_r2_index]
    if loss_val_fold != None:
        max_train_r2_val = r2_val_fold[max_train_r2_index]
        max_val_r2 = max(r2_val_fold)
        max_val_r2_index = r2_val_fold.index(max_val_r2)
        max_val_r2_train = r2_train_fold[max_val_r2_index]
        max_val_r2_test = r2_test_fold[max_val_r2_index]

    max_test_r2 = max(r2_test_fold)
    max_test_r2_index = r2_test_fold.index(max_test_r2)
    max_test_r2_train = r2_train_fold[max_test_r2_index]
    if loss_val_fold != None:
        max_test_r2_val = r2_val_fold[max_test_r2_index]

    if loss_val_fold != None:
        # 绘制水平红线标注每个数据集的最大值
        plt.axhline(y=max_train_r2, linestyle='--',
                label=f'Train Max R2--{max_train_r2_index} = ({max_train_r2:.3f}, {max_train_r2_val:.3f}, {max_train_r2_test:.3f})')

        plt.axhline(y=max_val_r2, color='orange', linestyle='--',
                label=f'Val Max R2--{max_val_r2_index} = ({max_val_r2_train:.3f}, {max_val_r2:.3f}, {max_val_r2_test:.3f})')
        plt.axhline(y=max_test_r2, color='green', linestyle='--',
                label=f'Test Max R2--{max_test_r2_index} = ({max_test_r2_train:.3f}, {max_test_r2_val:.3f}, {max_test_r2:.3f}')
    else:
        plt.axhline(y=max_train_r2, linestyle='--',
                    label=f'Train Max R2--{max_train_r2_index} = ({max_train_r2:.3f}, {max_train_r2_test:.3f})')
        plt.axhline(y=max_test_r2, color='green', linestyle='--',
                    label=f'Test Max R2--{max_test_r2_index} = ({max_test_r2_train:.3f}, {max_test_r2:.3f}')
    plt.legend()
    plt.grid(True)