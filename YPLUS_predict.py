import os
import numpy as np
from joblib import load

# 土地利用类别数量
lucc_num = 5

# 数据保存路径
root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 读取驱动因素数据
X_train = np.load(os.path.join(root_path, '2020drving_data.npy'))
Y_train = np.load(os.path.join(root_path, '2020lucc_data.npy'))
print('#'*6, 'X_train的形状', '#'*6)
print(X_train.shape)

# 剔除无效的数据
print('#'*6, '剔除无效的数据', '#'*6)
row_has_nan = np.any((X_train < -1e30) | (X_train > 1e30) | (X_train == 0), axis=0)
X_train = X_train[:, ~row_has_nan]
print('#'*6, 'X_train的形状', '#'*6)
print(X_train.shape)
print('#'*6, '保存无效值索引', '#'*6)
np.save(os.path.join(root_path, f'row_has_nan_2020.npy'), row_has_nan)

predictions = []
for i in range(1, lucc_num+1):
    rfr = load(os.path.join(root_path, f'rfr2020_{i}.joblib'))
    _predictions = rfr.predict(X_train.T)
    predictions.append(_predictions)
print('#' * 5, f'预测结果', '#' * 5)
predictions = np.array(predictions).T
print('#' * 5, f'数据形状', '#' * 5)
print(predictions.shape)
np.save(os.path.join(root_path, 'pred_rfr2020.npy'), predictions)
print('over')