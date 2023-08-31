'''
作者：卢沛安
时间：2023年08月31日
'''
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
from matplotlib.font_manager import FontProperties

# 指定中文字体文件路径
font_path = r'C:\Windows\Fonts\msyh.ttc'  # Microsoft YaHei 字体文件路径
# 加载中文字体
custom_font = FontProperties( fname=font_path )
# 配置Matplotlib使用中文字体
plt.rcParams[ 'font.family' ] = custom_font.get_name()

from global_utils import seed_everything

SEED = 22
seed_everything( SEED )

# 查看GPU是否可用
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu :
    print( 'CUDA is not available.  Training on CPU ...' )
else :
    print( 'CUDA is available!  Training on GPU ...' )

'''
数据处理
'''

# 读取数据
# 存放数据的路径

path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\data_source\round1_train_data_label"
soda_train = nc.Dataset( path + '\SODA_train.nc' )
soda_label = nc.Dataset( path + '\SODA_label.nc' )
cmip_train = nc.Dataset( path + '\CMIP_train.nc' )
cmip_label = nc.Dataset( path + '\CMIP_label.nc' )


# 数据扁平化
## 采用滑窗构造数据集
def flatten_features( train_ds , years , model_i , keys , start_idx ) :
    blocks = [ ]
    for key in keys :
        block = train_ds[ key ][ start_idx + model_i * years : start_idx + (model_i + 1) * years , :12 ].reshape( -1 ,
                                                                                                                  24 ,
                                                                                                                  72 ,
                                                                                                                  1 ).data
        blocks.append( block )
    return np.concatenate( blocks , axis=-1 )


def flatten_labels( label_ds , years , model_i , label_key , start_idx ) :
    return np.concatenate( [
        label_ds[ label_key ][ start_idx + model_i * years : start_idx + (model_i + 1) * years , 12 : 24 ].reshape(
            -1 ).data ,
        label_ds[ label_key ][ start_idx + (model_i + 1) * years - 1 , 24 : 36 ].reshape( -1 ).data
    ] , axis=0 )


def make_flatted( train_ds , label_ds , info , start_idx=0 ) :
    keys = [ 'sst' , 't300' , 'ua' , 'va' ]
    label_key = 'nino'
    years , models = info[ 1 ] , info[ 2 ]

    train_list = [ ]
    label_list = [ ]

    for model_i in range( models ) :
        train_flatted = flatten_features( train_ds , years , model_i , keys , start_idx )
        label_flatted = flatten_labels( label_ds , years , model_i , label_key , start_idx )

        train_list.append( train_flatted )
        label_list.append( label_flatted )

    return train_list , label_list


soda_info = ('soda' , 100 , 1)
cmip6_info = ('cmip6' , 151 , 15)
cmip5_info = ('cmip5' , 140 , 17)

soda_trains , soda_labels = make_flatted( soda_train , soda_label , soda_info )
cmip6_trains , cmip6_labels = make_flatted( cmip_train , cmip_label , cmip6_info )
cmip5_trains , cmip5_labels = make_flatted( cmip_train , cmip_label , cmip5_info , cmip6_info[ 1 ] * cmip6_info[ 2 ] )

# 得到扁平化后的数据维度为（模式数×序列长度×纬度×经度×特征数），其中序列长度=年数×12
print( np.shape( soda_trains ) , np.shape( cmip6_trains ) , np.shape( cmip5_trains ) )

# 填充SODA数据中的空值
soda_trains = np.array( soda_trains )
soda_trains_nan = np.isnan( soda_trains )
soda_trains[ soda_trains_nan ] = 0
print( 'Number of null in soda_trains after fillna:' , np.sum( np.isnan( soda_trains ) ) )

# 填充CMIP6数据中的空值
cmip6_trains = np.array( cmip6_trains )
cmip6_trains_nan = np.isnan( cmip6_trains )
cmip6_trains[ cmip6_trains_nan ] = 0
print( 'Number of null in cmip6_trains after fillna:' , np.sum( np.isnan( cmip6_trains ) ) )

# 填充CMIP5数据中的空值
cmip5_trains = np.array( cmip5_trains )
cmip5_trains_nan = np.isnan( cmip5_trains )
cmip5_trains[ cmip5_trains_nan ] = 0
print( 'Number of null in cmip6_trains after fillna:' , np.sum( np.isnan( cmip5_trains ) ) )

# 构造训练集

X_train = [ ]
y_train = [ ]
# 从CMIP5的17种模式中各抽取100条数据
for model_i in range( 17 ) :
    samples = np.random.choice( cmip5_trains.shape[ 1 ] - 12 , size=100 )
    for ind in samples :
        X_train.append( cmip5_trains[ model_i , ind : ind + 12 ] )
        y_train.append( cmip5_labels[ model_i ][ ind : ind + 24 ] )
# 从CMIP6的15种模式种各抽取100条数据
for model_i in range( 15 ) :
    samples = np.random.choice( cmip6_trains.shape[ 1 ] - 12 , size=100 )
    for ind in samples :
        X_train.append( cmip6_trains[ model_i , ind : ind + 12 ] )
        y_train.append( cmip6_labels[ model_i ][ ind : ind + 24 ] )
X_train = np.array( X_train )
y_train = np.array( y_train )

# 构造验证集

X_valid = [ ]
y_valid = [ ]
samples = np.random.choice( soda_trains.shape[ 1 ] - 12 , size=100 )
for ind in samples :
    X_valid.append( soda_trains[ 0 , ind : ind + 12 ] )
    y_valid.append( soda_labels[ 0 ][ ind : ind + 24 ] )
X_valid = np.array( X_valid )
y_valid = np.array( y_valid )

print( X_train.shape , y_train.shape , X_valid.shape , y_valid.shape )
np_data_path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\TCNN_RNN\np_data"
np.save( np_data_path + '\X_train_sample.npy' , X_train )
np.save( np_data_path + '\y_train_sample.npy' , y_train )
np.save( np_data_path + '\X_valid_sample.npy' , X_valid )
np.save( np_data_path + '\y_valid_sample.npy' , y_valid )


if __name__ == '__main__' :
    print( "finished!" )
