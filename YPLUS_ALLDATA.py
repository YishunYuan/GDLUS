import os
import numpy as np
from PIL import Image
from YPLUS_DATA_def import save_dataset



# 定义数据储存的目录
root_path = r'D:\Arcgis\PLUSProject'

# 定义需要读取的文件列表,土地利用分类放到最后
files = ['beds_num',
         'GDP',
         'municipal_revenues',
         'seg_rs_DEM',
         'seg_rs_Light',
         'seg_rs_NDVI',
         'seg_rs_People',
         'seg_rs_Pet',
         'seg_rs_Rain',
         'seg_rs_Temperature',
         'Slope',
         'Road',
         'students_num',
         'rc_seg_rs_GL30',]

# 定义起始年份
fy = 10

# 定义终止年份
ey = 20

# 定义需要读取的土地利用数据
save_dict= {'bed_num': [f'beds_num20{i}.tif' for i in range(fy, ey)],
            'gzgdp': [f'gzgdp20{i}.tif' for i in range(fy, ey)],
            'municipal_revenues': [f'municipal_revenues20{i}.tif' for i in range(fy, ey)],
            'PANDA': [f'seg_rs_PANDA_China_20{i}.tif' for i in range(fy, ey)],
            'NDVImax': [f'seg_rs_NDVImax20{i}.tif' for i in range(fy, ey)],
            'people': [f'seg_rs_广东省_chn_ppp_20{i}_UNadj.tif' for i in range(fy, ey)],
            'pet': [f'seg_rs_pet_{i}.tif' for i in range(fy, ey)],
            'rain': [f'seg_rs_R20{i}.tif' for i in range(fy, ey)],
            'temperature': [f'seg_rs_T20{i}.tif' for i in range(fy, ey)],
            'students_num': [f'students_num20{i}.tif' for i in range(fy, ey)],
            'lucc': [f'rc_seg_rs_CLCD_v01_20{i}_albert.tif' for i in range(fy, ey)],
            'other': ['seg_rs_dem.tif', 'slope.tif', 'railway.tif', 'trunk.tif', 'secondary.tif']}

# 定义储存数据的字典
file_dict = {}

print('#'*6, '读取数据', '#'*6)
for file in files:
    _files_path = os.path.join(root_path, file)
    _files_list = os.listdir(_files_path)
    for _file_name in _files_list:
        if _file_name[-3:] == "tif":
            print(_file_name)
            _file_image = Image.open(os.path.join(_files_path, _file_name))
            _file_array = np.array(_file_image)
            file_dict[_file_name] = _file_array

save_dataset(save_dict, r'D:\Project\DL_CA\YPLUS_DATA', file_dict)


