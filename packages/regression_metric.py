import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_pred, y_true):
    """
    计算回归任务的多个指标。

    参数:
    y_true (numpy.ndarray): 真实值。
    y_pred (numpy.ndarray): 预测值。

    返回:
    dict: 包含MSE, RMSE, MAE, MAPE, R², 和 Max Error的字典。
    """
    # 确保 y_true 和 y_pred 是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算均方误差（MSE）
    mse = np.mean((y_true - y_pred) ** 2)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)

    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(y_true - y_pred))

    # 计算平均绝对百分比误差（MAPE）
    mask = y_true != 0  # 避免除零
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # 计算R²分数（决定系数）
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 计算最大误差（Max Error）
    max_err = np.max(np.abs(y_true - y_pred))

    # 计算皮尔逊相关系数（PCC）
    pcc, _ = pearsonr(y_true, y_pred)

    # 计算斯皮尔曼相关系数（Spearman Coefficient）
    spearman, _ = spearmanr(y_true, y_pred)

    # 将所有指标放入字典中
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Max Error": max_err,
        "PCC": pcc,
        "Spearman Coefficient": spearman
    }

    return metrics