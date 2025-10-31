import numpy as np

# 定义用地需求和用地类型
Landuse_name = ['农田', '绿地', '水体', '裸地', '不透水面']
Landuse_demand = {1: 224914, 2: 327187, 3: 51073, 4: 66, 5: 110883}
class_num = 5
# 定义YPLUS参数
uk_threshold = 0.1
decay_step_number = 1
attenuation_factor = 0.5
percentage_of_seeds = 0.01
step = 500
threshold = 5
weight = [1, 1,  1, 1, 1]
transition_matrix = [[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 0, 1]]

# 读取数据
row_has_nan = np.load(r'D:\Project\DL_CA\YPLUS_DATA\row_has_nan.npy')
predictions = np.load('D:\Project\DL_CA\YPLUS_DATA\pred_rfr.npy')
lucc_data = np.load('D:\Project\DL_CA\YPLUS_DATA\lucc_data.npy')
limit_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2010limit_data.npy')

# 读取数据长度
eff_len = len(predictions)
print(eff_len)
# 进行随机选择
object = np.random.choice(eff_len, eff_len, replace=False)
# 定义土地利用类型的种类
array = lucc_data[1, :, :]
print(array.shape)
h, w = array.shape
land_use_len = len(Landuse_name)

# 提取其索引
indices = np.where(row_has_nan)
# 将对象中对应的元素打包成一个个元组, 然后返回由这些元组组成的列表
index_list = list(zip(indices[0], indices[1]))
print(len(index_list))

ca_seed = int(eff_len * percentage_of_seeds)
print('#' * 5, f'迭代用地数据长度', '#' * 5)
print(ca_seed)

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

# 创建初始惯性系数
intertial_list = list(np.ones(land_use_len))
print('#' * 5, f'intertial_list', '#' * 5)
print(intertial_list)
Gt = []
Gt_2 = []
landuse_list = []

stop = 0


for n in range(land_use_len):
    print('#' * 5, f'统计用地类型{n + 1}', '#' * 5)
    landuse = np.sum(array == n + 1)
    print(landuse)
    landuse_list.append(landuse)
    Gt.append(Landuse_demand[n + 1] - landuse)
    Gt_2.append(Landuse_demand[n + 1] - landuse)

print('#' * 5, 'Gt', '#' * 5)
print(Gt)

# 进行随机选择
intertial = 1
while intertial:
    print('#' * 5, f'迭代次数', '#' * 5)
    print(intertial)
    object_list = np.random.choice(eff_len, ca_seed, replace=False)
    for o in object_list:
        i, j = index_list[o]
        if 1 <= i <= h - 2 and 1 <= j <= w - 2:
            neighborhood = [array[x, y] for x in range(i - 1, i + 2) for y in range(j - 1, j + 2)]
        else:
            continue

        if 15 in neighborhood:
            continue

        o_landuse = neighborhood[4]
        neighborhood.pop(4)
        neighborhood = np.array(neighborhood)
        rand = np.random.rand(land_use_len)
        pr_list = [predictions[o][y] for y in range(land_use_len)]
        neighborhood_effects_list = []
        p_list = []
        for n in range(land_use_len):
            con = np.sum(neighborhood == n + 1)
            neighborhood_effects = con * weight[n] / 8
            if neighborhood_effects == 0 and rand[n] < pr_list[n]:
                neighborhood_effects = rand[n] * uk_threshold
            neighborhood_effects_list.append(neighborhood_effects)
            p_list.append(pr_list[n] * neighborhood_effects_list[n] * intertial_list[n])

        p_list = np.array(p_list)
        total_p = np.sum(p_list)
        cp_list = np.cumsum(p_list / total_p)
        rwc = np.random.rand(1)
        for index, cp in enumerate(cp_list):
            if rwc[0] <= cp:
                new_land_use = index + 1
                break
        r1 = generate_normal_with_range(1, 0.5, 0, 2)
        degradation_threshold = attenuation_factor ** decay_step_number * r1
        if not (Gt[new_land_use - 1] <= threshold) and Gt[new_land_use - 1] > 0:
            if transition_matrix[o_landuse - 1][new_land_use - 1] and limit_data[i, j] > 0:
                if pr_list[new_land_use - 1] > degradation_threshold:
                    array[i, j] = new_land_use
                    Gt[new_land_use - 1] -= 1
                    landuse_list[new_land_use - 1] += 1
                    Gt[o_landuse - 1] += 1
                    landuse_list[o_landuse - 1] -= 1


    print(landuse_list)
    print(intertial_list)

    if intertial >= 3:
        # print('#' * 5, f'Gt_1', '#' * 5)
        # print(Gt_1)
        # print('#' * 5, f'Gt_2', '#' * 5)
        # print(Gt_2)
        for n in range(land_use_len):
            if not (abs(Gt_1[n]) <= abs(Gt_2[n])):
                if Gt_2[n] < 0:
                    intertial_list[n] = intertial_list[n] * (abs(Gt_2[n]) + 1) / (abs(Gt_1[n]) + 1)
                else:
                    intertial_list[n] = intertial_list[n] * (abs(Gt_1[n]) + 1) / (abs(Gt_2[n]) + 1)
            else:
                intertial_list[n] = 1

    if intertial >= 2:
        stop_list = [True for d in Gt if abs(d) < threshold]
        stop = sum(stop_list)
        if stop == land_use_len or (Gt == Gt_1 == Gt_2):
            intertial = -1
        Gt_2 = Gt_1.copy()
        if sum(map(abs, Gt_1)) - sum(map(abs, Gt)) < step:
            decay_step_number += 1
    Gt_1 = Gt.copy()
    intertial += 1

np.save(r'D:\Project\DL_CA\YPLUS_DATA\frfr_p.npy', array)
print('over')