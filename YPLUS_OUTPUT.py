import os
import shutil
import rasterio
import numpy as np

save_path = r'D:\Project\DL_CA\new_data\20CLULC\new_result1.tif'
model_file = r'D:\Arcgis\PLUSProject\nrc_nseg_nrsGL30\nrc_nseg_nrsCLCD_v01_2011_albert.tif'
output_array = np.load(r'D:\Project\DL_CA\YPLUS_DATA\nf1.npy')
shutil.copy(model_file, save_path)

with rasterio.open(model_file) as src:
                # 读取地理空间坐标信息
                spatial_info = src.profile
                image_data = src.read()
                image_data = np.array(image_data)
                # print(image_data.shape)
                # 读取图像数据
# 保存 TIFF 图片
with rasterio.open(save_path, 'w', **spatial_info) as dst:
    # 写入图像数据
    dst.write(output_array[np.newaxis, :, :])