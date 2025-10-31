import os.path

import numpy as np

def save(filename, save_name, data):
    print('#'*6, f'保存{filename}数据', '#'*6)
    print(f'保存路径{save_name}')
    np.save(save_name, data)

def savenpy(filename, file_list, file_dict, save_name):
    print('#'*6, f'堆叠{filename}数据', '#'*6)
    for i in range(len(file_list)-1):
        if not i:
            print(f'正在堆叠{file_list[i]}和{file_list[i+1]}')
            data = np.stack((file_dict[file_list[i]], file_dict[file_list[i+1]]), axis=0)
        else:
            print(f'正在堆叠{file_list[i+1]}')
            data = np.concatenate((data, file_dict[file_list[i+1]][np.newaxis, :, :]), axis=0)

    print('#' * 6, f'{filename}数据形状', '#' * 6)
    print(data.shape)

    save(filename, save_name, data)
def save_dataset(save_dict, save_name, file_dict):
    for k in save_dict.keys():
        savenpy(k, save_dict[k], file_dict, os.path.join(save_name, k))



