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

# 定义需要读取的驱动因子
drivingdata_list = ['nseg_nrsbeds_num2020.tif',
                    'nseg_nrsgzgdp2020.tif',
                    'nseg_nrsmunicipal_revenues2020.tif',
                    'nseg_nrsdem.tif',
                    'nseg_nrsPANDA_China_2020.tif',
                    'nseg_nrsNDVImax2020.tif',
                    'nseg_nrs广东省_chn_ppp_2020_UNadj.tif',
                    'nseg_nrspet_20.tif',
                    'nseg_nrsR2020.tif',
                    'nseg_nrsT2020.tif',
                    'slope.tif',
                    'nseg_nrsrailway.tif',
                    'nseg_nrssecondary.tif',
                    'nseg_nrstrunk.tif',
                    'nseg_nrsstudents_num2020.tif']

# 定义需要读取的土地利用数据
lucc_list = ['nrc_nseg_nrsCLCD_v01_2000_albert.tif',
             'nrc_nseg_nrsCLCD_v01_2020_albert.tif']

limit_name = 'limit_nrc_nseg_nrsCLCD_v01_2020_albert.tif'


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

savenpy('驱动因子', drivingdata_list, file_dict,r'D:\Project\DL_CA\YPLUS_DATA\2020drving_data.npy')
savenpy('土地利用', lucc_list, file_dict,r'D:\Project\DL_CA\YPLUS_DATA\2020lucc_data.npy')
np.save(r'D:\Project\DL_CA\YPLUS_DATA\2020limit_data.npy', file_dict[limit_name])


