'''
作者：卢沛安
时间：2023年08月28日
'''
from typing import List , Tuple , Dict , Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定中文字体文件路径
font_path = r'C:\Windows\Fonts\msyh.ttc'  # Microsoft YaHei 字体文件路径
# 加载中文字体
custom_font = FontProperties( fname=font_path )
# 配置Matplotlib使用中文字体
plt.rcParams[ 'font.family' ] = custom_font.get_name()

import netCDF4 as nc
from netCDF4 import Dataset

test = Dataset( 'test.nc' , 'w' , 'NETCDF4' )
# - 打开已存在的netCDF文件
# 打开训练样本中的SODA数据
soda = Dataset( r"D:\DATA\ENSO\enso_round1_train_20210201\SODA_train.nc" )
print('SODA文件格式：', soda.data_model)
# 查看文件中包含的对象
print(soda)
print(soda.dimensions)
print(soda.variables)

# 读取每个变量的数据
soda_sst = soda['sst'][:]
print(soda_sst[1, 1, 1, 1])

soda_t300 = soda['t300'][:]
print(soda_t300[1, 2, 12:24, 36])

soda_ua = soda['ua'][:]
print(soda_ua[1, 2, 12:24:2, 36:38])

soda_va = soda['va'][:]
print(soda_va[5:10, 0:12, 12, 36])



if __name__ == '__main__' :
    print( "finished!" )
