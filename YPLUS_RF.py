import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

# 土地利用类别数量
lucc_num = 5

# 数据保存路径
root_path = r'D:/Project/DL_CA/YPLUS_DATA/'

# 读取数据
X_train = np.load('D:\Project\DL_CA\YPLUS_DATA\X_train_sample_20200.npy')
Y_train = np.load('D:\Project\DL_CA\YPLUS_DATA\Y_train_sample_2020.npy')
print('#'*6, 'X_train的形状', '#'*6)
print(X_train.shape)

# 贡献度
contribution_dict = {'feature': ['beds_num',
                                 'gdp',
                                 'municipal_revenues',
                                 'dem',
                                 'light',
                                 'NDVI',
                                 'people',
                                 'pet',
                                 'rain',
                                 'temperature',
                                 'slope',
                                 'railway',
                                 'secondary',
                                 'trunk',
                                 'students_num']}

for i in range(1, lucc_num+1):
    print('#'*6, f'计算第{i}种用地类型', '#'*6)
    diff = Y_train == i
    indices = np.where(diff)[0]
    _Y_train = np.array(diff, dtype=np.int8)

    rfr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, max_features=X_train.shape[0])
    rfr.fit(X_train.T, _Y_train)

    feature_import = rfr.feature_importances_
    contribution_dict[f'土地利用{i}'] = feature_import

    # 将特征名称和重要性组合成一个DataFrame
    feature_importance_df = pd.DataFrame(contribution_dict)

    dump(rfr, os.path.join(os.path.join(root_path, f'rfr2020_{i}.joblib')))

# 显示贡献度
print("#" * 5, "贡献度数据", "#" * 5)
print(feature_importance_df)
# 将贡献度保存到excel文件中
feature_importance_df.to_excel(os.path.join(root_path, f'feature_importance_rf2020.xlsx'))