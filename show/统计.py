import numpy as np
import pandas as pd
import seaborn as sns
from graph_emb import load_pre

file_fold = f'../save_result/real/HGCN_res1/PoincareBall/2025_08_13_15_15/'
# file_fold = f'../save_result/real/GCN/Hyperboloid/2025_08_20_12_07/'

pre_couple_file_test = file_fold+'pre/test/0.csv'
couple, pre_test, true_test = load_pre(pre_couple_file_test)
# 转成numpy array（如果还不是）
pre_test = np.array(pre_test)
true_test = np.array(true_test)
couple = np.array([c[0]+'|'+c[1] if isinstance(c, (list, tuple)) else c for c in couple])

# 计算残差
abs_error = np.abs(pre_test - true_test)

# 获取最小误差的索引（升序）
best_indices = np.argsort(abs_error)[:1000]

# 提取对应的信息
best_couples = couple[best_indices]
best_preds = pre_test[best_indices]
best_trues = true_test[best_indices]
best_errors = np.array(abs_error[best_indices])

# 整理成DataFrame
df_best = pd.DataFrame({
    'Protein Pair': best_couples,
    'True ΔTm': best_trues,
    'Predicted ΔTm': best_preds,
    'Absolute Error': best_errors
})
print(df_best)
# # 展示结果
# import caas_jupyter_tools as cjtools
# cjtools.display_dataframe_to_user(name="Top 10 Best Predicted Pairs", dataframe=df_best)