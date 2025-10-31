import os
import sys
import numpy as np

# 定义数据储存的目录
root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 定义用地需求和用地类型
class_num = 5
Landuse_name = ['农田', '绿地', '水体', '裸地', '不透水面']
Landuse_demand = {1: 224914, 2: 327187, 3: 51073, 4: 66, 5: 110883}

# 定义YPLUS参数
uk_threshold = 2.0
decay_step_number = 1
attenuation_factor = 0.9
step = 500
changenum = 100
threshold = 2
weight = [1, 1,  1, 1, 1]
transition_matrix = [[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]]

# 读取数据
row_has_nan = np.load(r'D:\Project\DL_CA\YPLUS_DATA\row_has_nan.npy')
predictions = np.load('D:\Project\DL_CA\YPLUS_DATA\pred_rfr.npy')
lucc_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2010lucc_data.npy')
limit_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2010limit_data.npy')


# 读取数据长度
eff_len = len(predictions)
print(eff_len)
# 进行随机选择
object = np.random.choice(eff_len, eff_len, replace=False)
# 定义土地利用类型的种类
array = lucc_data[0, :, :]
print(array.shape)
h, w = array.shape
land_use_len = len(Landuse_name)

# 提取其索引
indices = np.where(~row_has_nan)
# 将对象中对应的元素打包成一个个元组, 然后返回由这些元组组成的列表
index_list = list(zip(indices[0], indices[1]))
print(len(index_list))

# 定义
def generate_normal_with_range(mean, std_dev, lower, upper, size=1):
    values = []
    while len(values) < size:
        v = np.random.normal(mean, std_dev)
        if lower <= v <= upper:
            values.append(v)
    if size == 1:
        return values[0]
    else:
        return np.array(values)

def sysexit(Gt_1):
    stop_list = [1 for d in Gt_1 if abs(d) <= threshold]
    stop = sum(stop_list)
    if stop == land_use_len:
        np.save(os.path.join(root_path, 'fs_rfr.npy'), array)
        print('over')
        sys.exit()

intertial_list = list(np.ones(land_use_len))
print('#' * 5, f'intertial_list', '#' * 5)
print(intertial_list)
Gt_2 = []
Gt_1 = []
landuse_list = []
stop = 0
_step = step
for n in range(land_use_len):
    print('#' * 5, f'统计用地类型{n + 1}', '#' * 5)
    landuse = np.sum(array == n + 1)
    print(landuse)
    landuse_list.append(landuse)
    Gt_1.append(Landuse_demand[n + 1] - landuse)
    Gt_2.append(Landuse_demand[n + 1] - landuse)
print('#' * 5, 'Gt_1', '#' * 5)
print(Gt_1)
for o in object:
    # print('#' * 5, '目标', '#' * 5)
    # print(o)
    i, j = index_list[o]
    # print('#' * 5, '坐标', '#' * 5)
    # print(i,j)
    if 1 <= i <= h - 2 and 1 <= j <= w - 2:
        neighborhood = [array[x, y] for x in range(i - 1, i + 2) for y in range(j - 1, j + 2)]
    else:
        continue

    if 15 in neighborhood:
        continue

    o_landuse = neighborhood[4]
    neighborhood.pop(4)
    neighborhood = np.array(neighborhood)
    # print('#' * 5, f'领域', '#' * 5)
    # print(neighborhood)
    rand = np.random.rand(land_use_len)
    # print('#' * 5, f'random patch', '#' * 5)
    # print(rand)
    pr_list = [predictions[o][y] for y in range(land_use_len)]
    # print('#' * 5, f'pr_list', '#' * 5)
    # print(pr_list)
    neighborhood_effects_list = []
    p_list = []
    for n in range(land_use_len):
        con = np.sum(neighborhood == n + 1)
        # print('#' * 5, f'con', '#' * 5)
        # print(con)
        neighborhood_effects = con * weight[n] / 8
        if neighborhood_effects == 0:
            neighborhood_effects = rand[n] * uk_threshold
        neighborhood_effects_list.append(neighborhood_effects)
    # print('#' * 5, f'neighborhood_effects_list', '#' * 5)
    # print(neighborhood_effects_list)
    # print('#' * 5, f'Gt_1', '#' * 5)
    # print(Gt_1)
    for n in range(land_use_len):
        if not (abs(Gt_1[n]) <= abs(Gt_2[n])):
            if Gt_2[n] < 0:
                intertial_list[n] = intertial_list[n] * Gt_2[n] / Gt_1[n]
            else:
                intertial_list[n] = intertial_list[n] * Gt_1[n] / Gt_2[n]
        p_list.append(pr_list[n] * neighborhood_effects_list[n] * intertial_list[n])
    # print('#' * 5, f'p_list', '#' * 5)
    # print(p_list)
    p_list = np.array(p_list)
    total_p = np.sum(p_list)
    cp_list = np.cumsum(p_list / total_p)
    # print('#' * 5, 'cp_list', '#' * 5)
    # print(cp_list)
    counter = 20
    while counter:
        rwc = np.random.rand(1)
        # print('#' * 5, 'rwc', '#' * 5)
        # print(rwc)
        for index, cp in enumerate(cp_list):
            if rwc[0] <= cp:
                new_land_use = index + 1
                break

        if counter <= 10:
            new_land_use = np.random.randint(low=0, high=land_use_len, size=1)[0] + 1
        counter -= 1

        r1 = generate_normal_with_range(1, 0.5, 0, 2)
        degradation_threshold = attenuation_factor ** decay_step_number * r1
        if not(abs(Gt_1[new_land_use - 1]) == 0) and Gt_1[new_land_use - 1] >= 0:
            if transition_matrix[o_landuse - 1][new_land_use - 1] and limit_data[i, j] > 0:
                if p_list[new_land_use - 1] > degradation_threshold:
                    array[i, j] = new_land_use
                    Gt_1[new_land_use - 1] -= 1
                    landuse_list[new_land_use - 1] += 1
                    Gt_1[o_landuse - 1] += 1
                    landuse_list[o_landuse - 1] -= 1
                    print(landuse_list)
        else:
            sysexit(Gt_1)
            new_land_use = 0

        if new_land_use:
            break

    if _step == 0:
        decay_step_number += 1
        _step = step
    _step -= 1

    if _step % changenum == 0:
        Gt_2 = Gt_1
        
    sysexit(Gt_1)

np.save(os.path.join(root_path, 'frfr.npy'), array)
print('over')