import os
import numpy as np
from PIL import Image
from YPLUS_DATA_def import savenpy

# 定义数据储存的目录
root_path = r'D:\Arcgis\PLUSProject'

# 定义需要读取的文件列表,土地利用分类放到最后
files = ['nseg_nrsbeds_num',
         'nseg_nrsGDP',
         'nseg_nrsmunicipal_revenues',
         'nseg_nrsDEM',
         'nseg_nrsLight',
         'nseg_nrsNDVI',
         'nseg_nrsPeople',
         'nseg_nrsPet',
         'nseg_nrsRain',
         'nseg_nrsTemperature',
         'nseg_nrsSlope',
         'nseg_nrsRoad',
         'nseg_nrsstudents_num',
         'limit_nrc_nseg_nrsGL30',
         'nrc_nseg_nrsGL30',]

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


for i in range(10):
    year = i
    # 定义需要读取的驱动因子
    drivingdata_list = [f'nseg_nrsbeds_num201{year}.tif',
                        f'nseg_nrsgzgdp201{year}.tif',
                        f'nseg_nrsmunicipal_revenues201{year}.tif',
                        'nseg_nrsdem.tif',
                        f'nseg_nrsPANDA_China_201{year}.tif',
                        f'nseg_nrsNDVImax201{year}.tif',
                        f'nseg_nrs广东省_chn_ppp_201{year}_UNadj.tif',
                        f'nseg_nrspet_1{year}.tif',
                        f'nseg_nrsR201{year}.tif',
                        f'nseg_nrsT201{year}.tif',
                        'slope.tif',
                        'nseg_nrsrailway.tif',
                        'nseg_nrssecondary.tif',
                        'nseg_nrstrunk.tif',
                        f'nseg_nrsstudents_num201{year}.tif']
    # 定义需要读取的土地利用数据
    if year != 9:
        lucc_list = [f'nrc_nseg_nrsCLCD_v01_201{year}_albert.tif',
                     f'nrc_nseg_nrsCLCD_v01_201{year+1}_albert.tif']
        savenpy('土地利用', lucc_list, file_dict, r'D:\Project\DL_CA\YPLUS_DATA\201{}lucc_data.npy'.format(year))

    savenpy('驱动因子', drivingdata_list, file_dict,r'D:\Project\DL_CA\YPLUS_DATA\201{}drving_data.npy'.format(year))




