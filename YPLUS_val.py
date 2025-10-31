import numpy as np

def calculate(matrix):
    num_classes = matrix.shape[0]
    total = np.sum(matrix)
    accuracy = np.trace(matrix) / total
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    pe = 0
    for i in range(num_classes):
        # 避免除以0的错误
        column_sum = np.sum(matrix[:, i])
        row_sum = np.sum(matrix[i, :])
        recall[i] = matrix[i, i] / column_sum if column_sum != 0 else 0
        precision[i] = matrix[i, i] / row_sum if row_sum != 0 else 0
        pe += column_sum * row_sum
    pe = pe / total ** 2
    kappaa = (accuracy - pe) / (1 - pe)
    return accuracy, kappaa, precision, recall

# 定义数据储存的目录
val_root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 定义采样率
sampling_num = 1

for i in range(4):
    globals()[chr(ord('A') + i)] = 0

Landuse_name = ['农田', '绿地', '水体', '裸地', '不透水面']
lucc_num = len(Landuse_name)
confusion_matrix = np.zeros((lucc_num, lucc_num))
lucc_data = np.load(r'D:\Project\DL_CA\YPLUS_DATA\2010lucc_data.npy')
simulation = np.load(r'D:\Project\DL_CA\YPLUS_DATA\frfr.npy')
h, w = simulation.shape
x = np.random.randint(low=0, high=h, size=int((h * w * sampling_num)**0.5))
y = np.random.randint(low=0, high=w, size=int((h * w * sampling_num)**0.5))

for i in x:
    for j in y:
        print(i, j)
        result = lucc_data[1, i, j]
        old = lucc_data[0, i, j]
        future = simulation[i, j]
        if result != 15 and old != 15 and future != 15:
            confusion_matrix[result - 1, future - 1] += 1
            if future != old:
                if result == future:
                    B += 1
                elif result == old:
                    A += 1
                else:
                    C += 1
            else:
                if result != future:
                    D += 1
fom = B / (A + B + C + D)
producer_s_Accuracy = B / (A + B + C)
user_s_Accuracy = B / (B + C + D)

fom_dict = {'A':[A], 'B':[B], 'C':[C], 'D':[D], 'FOM':[fom], "producer's Accuracy":[producer_s_Accuracy], "user's Accuracy":[user_s_Accuracy]}
print(fom_dict)
accuracy, kappaa, precision, recall = calculate(confusion_matrix)
print(confusion_matrix)
print(accuracy)
print(kappaa)
print(precision)