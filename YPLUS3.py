import os
import numpy as np

# 定义数据储存的目录
root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 定义用地需求和用地类型
Landuse_name = ['农田', '绿地', '水体', '裸地', '不透水面']
Landuse_demand = np.array([219438, 321268, 39436, 153, 133828])
land_use_len = len(Landuse_name)

# 定义YPLUS参数
neighborhood_size = 7      # 领域尺寸，必须是大于等于3的奇数
uk_threshold = 0.1
max_iteration_number = 60
decay_step_number = 1
attenuation_factor = 0.5
percentage_of_seeds = 0.01
step = 500
threshold = 5
transition_matrix = [[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 0, 1]]

# 读取数据
row_has_nan = np.load(r'D:\Project\DL_CA\YPLUS_DATA\row_has_nan_2020.npy')
predictions = np.load('D:\Project\DL_CA\YPLUS_DATA\pred_rfr2020.npy')
lucc_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2020lucc_data.npy')
limit_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2020limit_data.npy')

# 读取数据长度
eff_len = len(predictions)
print('数据长度:', eff_len)

# 定义土地利用类型的种类
lucc_array = lucc_data[0, :, :]
h, w = lucc_array.shape

# 提取其索引
indices = np.where(~row_has_nan)
# 将对象中对应的元素打包成一个个元组, 然后返回由这些元组组成的列表
index_list = list(zip(indices[0], indices[1]))
print('索引长度', len(index_list))

# 统计用地数量
landuse_indices = lucc_array != 15
landuse_list = np.bincount(lucc_array[landuse_indices].flatten(), minlength=land_use_len)
landuse_list = np.delete(landuse_list, 0, axis=0)
print('#' * 5, f'开始年份的土地利用数据', '#' * 5)
print(landuse_list)

# 自适应惯性系数
def adaptive_inertia_weight_coefficien(Driv, Dt_1, Dt_2):
    for n in range(land_use_len):
        if not (abs(Dt_1[n]) <= abs(Dt_2[n])):
            if Dt_2[n] < 0:
                Driv[n] = Driv[n] * (Dt_2[n] + 1) / (Dt_1[n] + 1)
            else:
                Driv[n] = Driv[n] * (Dt_1[n] + 1) / (Dt_2[n] + 1)
    print('Driv:', Driv)
    return np.abs(Driv)


Driv = np.ones(land_use_len)
Dt = Landuse_demand - landuse_list

# 统计迭代数量
Deviation = np.sum(np.abs(Dt))
dDeviation = -Deviation

# 领域计算函数
def neighborhood_weight(x, y, size=neighborhood_size, w=w, h=h):
    size = (size - 1) / 2
    neighborhood = np.array([], dtype=np.int8)
    for i in range(int(x - size), int(x + size)):
        for j in range(int(y - size), int(y + size)):
            if (i >= 0) and (i <= h) and (j >= 0) and (j <= w) and (lucc_array[i, j] != 15) and (i != x) and (j != y):
                neighborhood = np.append(neighborhood, lucc_array[i, j])
    neighborhood = np.bincount(neighborhood, minlength=land_use_len + 1)
    neighborhood = np.delete(neighborhood, 0, axis=0)
    return neighborhood / np.sum(neighborhood)

# 随机补丁种子模块
def random_patch_seeds(o, neighbor, Driv):
    r = np.random.rand(1)
    P = np.array([predictions[o, y] for y in range(land_use_len)])
    for n in range(land_use_len):
        if neighbor[n] == 0 and r < P[n]:
            neighbor[n] = r[0] * uk_threshold
    OP = P * neighbor * Driv
    return OP

# 轮盘赌算法
def roulette(OP):
    total = np.sum(OP)
    cp = np.cumsum(OP / total)
    rwc = np.random.rand(1)
    for i, c in enumerate(cp):
        if rwc <= c:
            new_land_use = i + 1
            return new_land_use

# 高斯分布
def normal_distribution():
    r1 = 0
    while not(0 < r1 < 2):
        r1 = np.random.normal(size=1, loc=1, scale=0.3)
    return r1

# CA迭代算法
count_pixel = 0
while (dDeviation != 0) and (count_pixel <= max_iteration_number):
    print('#' * 5, f'迭代次数', '#' * 5)
    print('count_pixel:', count_pixel)
    if count_pixel:
        NewDeviation = np.sum(np.abs(Landuse_demand - landuse_list))
        dDeviation = Deviation - NewDeviation
        Deviation = NewDeviation
    print('#' * 5, f'迭代对象', '#' * 5)
    print('Deviation:', Deviation)
    print('dDeviation:', dDeviation)
    object_list = np.random.choice(eff_len, int(h * w * percentage_of_seeds), replace=False)
    for o in object_list:
        i, j = index_list[o]
        old_land_use = lucc_array[i, j]
        # print(old_land_use)
        neighborhood = neighborhood_weight(i, j)
        # print('nw:', neighborhood)
        OP = random_patch_seeds(o, neighborhood, Driv)
        # print('OP:', OP)
        if np.all(OP==0):continue
        new_land_use = roulette(OP)
        # print('newlanduse:', new_land_use)
        r1 = normal_distribution()
        degradation_threshold = attenuation_factor ** decay_step_number * r1
        if Dt[new_land_use - 1]:
            if transition_matrix[old_land_use - 1][new_land_use - 1] and limit_data[i, j] > 0:
                if OP[new_land_use - 1] > degradation_threshold:
                    lucc_array[i, j] = new_land_use
                    # print('变化')
                    landuse_list[new_land_use - 1] += 1
                    landuse_list[old_land_use - 1] -= 1

    print('#' * 5, f'土地利用数据', '#' * 5)
    print(landuse_list)

    # 更新适应参数
    if count_pixel >= 1:
        Dt_2 = np.copy(Dt_1)
    Dt_1 = np.copy(Dt)
    Dt = Landuse_demand - landuse_list
    print('Dt:', Dt)

    # 更新衰减步数
    if np.sum(np.abs(Dt - 1)) - np.sum(np.abs(Dt)) < step:
        decay_step_number += 1

    if count_pixel >= 1:
        Driv = adaptive_inertia_weight_coefficien(Driv, Dt_1, Dt_2)

    # 迭代次数加一
    count_pixel += 1

np.save(os.path.join(root_path, 'nf1.npy'), lucc_array)
print('预测完成')



