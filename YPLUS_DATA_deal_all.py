import os
import numpy as np

# 土地利用类别数量
lucc_num = 5

# 采样率
sampling_rate = 0.9

# 数据保存路径
root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 读取土地利用数据
lucc_data = np.load(os.path.join(root_path, '2020lucc_data.npy'))

# 读取驱动因素数据
driving_data = np.load(os.path.join(root_path, '2020drving_data.npy'))

# 计算两年土地利用中变化的部分
diff = lucc_data[1, :, :] != lucc_data[0, :, :]
indices = np.where(diff)
np.save('D:\Project\DL_CA\YPLUS_DATA\expand_indices_2020.npy', indices)

print('#'*6, '读取数据', '#'*6)
Y_train = lucc_data[1, :, :][indices]
before_lucc_data = lucc_data[0, :, :][indices]
print('#'*6, 'Y_train的形状', '#'*6)
print(Y_train.shape)

X_train = np.stack((driving_data[0, :, :][indices], driving_data[1, :, :][indices]), axis=0)
for i in range(2, driving_data.shape[0]):
    X_train = np.concatenate((X_train, driving_data[i, :, :][indices][np.newaxis, :]), axis=0)
print('#'*6, 'X_train的形状', '#'*6)
print(X_train.shape)

counts = np.bincount(Y_train, minlength=lucc_num)
counts = np.delete(counts, 0, axis=0)
print('#'*6, '统计Y_train中各土地利用类别的数量', '#'*6)
print(counts)



transition_matrix = np.diag([1 for i in range(lucc_num)])
for i, j in zip(before_lucc_data, Y_train):
    transition_matrix[i-1, j-1] += 1

print('#'*6, '参数transition_matrix', '#'*6)
print(transition_matrix)

print('#'*6, '参数weight', '#'*6)
t_a = np.sum(transition_matrix, axis=0)
t_b = np.sum(transition_matrix, axis=1)
weight = t_a - t_b
weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
print(weight)

# 剔除无效的数据
print('#'*6, '剔除无效的数据', '#'*6)
row_has_nan = np.any((X_train < -1e30) | (X_train > 1e30) | (X_train == 0), axis=0)
X_train = X_train[:, ~row_has_nan]
Y_train = Y_train[~row_has_nan]
print('#'*6, 'X_train的形状', '#'*6)
print(X_train.shape)
print('#'*6, 'Y_train的形状', '#'*6)
print(Y_train.shape)

# 对数据进行随机采样
object = np.random.choice(Y_train.shape[0], int(Y_train.shape[0] * sampling_rate), replace=False)
print(object.shape)

Y_train = Y_train[object]
X_train = X_train[:, object]
print('#'*6, '采样后Y_train的形状', '#'*6)
print(Y_train.shape)
print('#'*6, '采样后X_train的形状', '#'*6)
print(X_train.shape)

counts = np.bincount(Y_train, minlength=lucc_num)
counts = np.delete(counts, 0, axis=0)
print('#'*6, '采样后Y_train中各土地利用类别的数量', '#'*6)
print(counts)

print('#'*6, '保存采样后的X_train和Y_train', '#'*6)
np.save('D:\Project\DL_CA\YPLUS_DATA\X_train_sample_2020.npy', X_train)
np.save('D:\Project\DL_CA\YPLUS_DATA\Y_train_sample_2020.npy', Y_train)