'''
作者：卢沛安
时间：2023年08月28日
'''
import math

import netCDF4 as nc
import numpy as np
import seaborn as sns

color = sns.color_palette()
sns.set_style( 'darkgrid' )
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from global_utils import seed_everything

# 指定中文字体文件路径
font_path = r'C:\Windows\Fonts\msyh.ttc'  # Microsoft YaHei 字体文件路径
# 加载中文字体
custom_font = FontProperties( fname=font_path )
# 配置Matplotlib使用中文字体
plt.rcParams[ 'font.family' ] = custom_font.get_name()

# 固定随机种子
SEED = 22

seed_everything( SEED )
path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\data_source\round1_train_data_label"
soda_train = nc.Dataset( path + '\SODA_train.nc' )
soda_label = nc.Dataset( path + '\SODA_label.nc' )
cmip_train = nc.Dataset( path + '\CMIP_train.nc' )
cmip_label = nc.Dataset( path + '\CMIP_label.nc' )

months = range( 0 , 12 )
month_labels = [ 'Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sept' , 'Oct' , 'Nov' , 'Dec' ]
# sin月份特征
months_sin = map( lambda x : math.sin( 2 * math.pi * x / len( months ) ) , months )
# cos月份特征
months_cos = map( lambda x : math.cos( 2 * math.pi * x / len( months ) ) , months )

# 绘制每个月的月份特征组合
plt.figure( figsize=(20 , 5) )
x_axis = np.arange( -1 , 13 , 1e-2 )
sns.lineplot( x=x_axis , y=np.sin( 2 * math.pi * x_axis / len( months ) ) )
sns.lineplot( x=x_axis , y=np.cos( 2 * math.pi * x_axis / len( months ) ) )
sns.scatterplot( x=months , y=months_sin , s=200 )
sns.scatterplot( x=months , y=months_cos , s=200 )
plt.xticks( ticks=months , labels=month_labels )
plt.show( block=True )

from numba import jit


@jit( nopython=True )
def fill_soda_matrix() :
    soda_month_sin = np.zeros( (100 , 36 , 24 , 72) )
    soda_month_cos = np.zeros( (100 , 36 , 24 , 72) )
    for y in range( 100 ) :
        for m in range( 36 ) :
            val_sin = np.sin( 2 * np.pi * (m % 12) / 12 )
            val_cos = np.cos( 2 * np.pi * (m % 12) / 12 )
            for lat in range( 24 ) :
                for lon in range( 72 ) :
                    soda_month_sin[ y , m , lat , lon ] = val_sin
                    soda_month_cos[ y , m , lat , lon ] = val_cos
    return soda_month_sin , soda_month_cos


@jit( nopython=True )
def fill_cmip_matrix() :
    cmip_month_sin = np.zeros( (4645 , 36 , 24 , 72) )
    cmip_month_cos = np.zeros( (4645 , 36 , 24 , 72) )
    for y in range( 4645 ) :
        for m in range( 36 ) :
            val_sin = np.sin( 2 * np.pi * (m % 12) / 12 )
            val_cos = np.cos( 2 * np.pi * (m % 12) / 12 )
            for lat in range( 24 ) :
                for lon in range( 72 ) :
                    cmip_month_sin[ y , m , lat , lon ] = val_sin
                    cmip_month_cos[ y , m , lat , lon ] = val_cos
    return cmip_month_sin , cmip_month_cos


soda_month_sin , soda_month_cos = fill_soda_matrix()
cmip_month_sin , cmip_month_cos = fill_cmip_matrix()

# 构造一个维度为100*36*24*72的矩阵，矩阵中的每个值为所在月份的sin函数值
# soda_month_sin = np.zeros( (100 , 36 , 24 , 72) )
# for y in range( 100 ) :
#     for m in range( 36 ) :
#         for lat in range( 24 ) :
#             for lon in range( 72 ) :
#                 soda_month_sin[ y , m , lat , lon ] = math.sin( 2 * math.pi * (m % 12) / 12 )
# assert soda_month_sin.all() == soda_month_sin_jit.all()

'''
数据扁平化
'''


# 数据扁平化
def make_flatted( train_ds , label_ds , month_sin , month_cos , info , start_idx=0 ) :
    keys = [ 'sst' , 't300' , 'ua' , 'va' ]
    label_key = 'nino'
    # 年数
    years = info[ 1 ]
    # 模式数
    models = info[ 2 ]

    train_list = [ ]
    label_list = [ ]

    # 将同种模式下的数据拼接起来
    for model_i in range( models ) :
        blocks = [ ]

        # 对每个特征，取每条数据的前12个月进行拼接
        for key in keys :
            block = train_ds[ key ][ start_idx + model_i * years : start_idx + (model_i + 1) * years , :12 ].reshape(
                -1 , 24 , 72 , 1 ).data
            blocks.append( block )
        # 增加sin月份特征
        block_sin = month_sin[ start_idx + model_i * years : start_idx + (model_i + 1) * years , :12 ].reshape( -1 ,
                                                                                                                24 ,
                                                                                                                72 , 1 )
        blocks.append( block_sin )
        # 增加cos月份特征
        block_cos = month_cos[ start_idx + model_i * years : start_idx + (model_i + 1) * years , :12 ].reshape( -1 ,
                                                                                                                24 ,
                                                                                                                72 , 1 )
        blocks.append( block_cos )

        # 将所有特征在最后一个维度上拼接起来
        train_flatted = np.concatenate( blocks , axis=-1 )

        # 取12-23月的标签进行拼接，注意加上最后一年的最后12个月的标签（与最后一年12-23月的标签共同构成最后一年前12个月的预测目标）
        label_flatted = np.concatenate( [
            label_ds[ label_key ][ start_idx + model_i * years : start_idx + (model_i + 1) * years , 12 : 24 ].reshape(
                -1 ).data ,
            label_ds[ label_key ][ start_idx + (model_i + 1) * years - 1 , 24 : 36 ].reshape( -1 ).data
        ] , axis=0 )

        train_list.append( train_flatted )
        label_list.append( label_flatted )

    return train_list , label_list


soda_info = ('soda' , 100 , 1)
cmip6_info = ('cmip6' , 151 , 15)
cmip5_info = ('cmip5' , 140 , 17)

soda_trains , soda_labels = make_flatted( soda_train , soda_label , soda_month_sin , soda_month_cos , soda_info )
cmip6_trains , cmip6_labels = make_flatted( cmip_train , cmip_label , cmip_month_sin , cmip_month_cos , cmip6_info )
cmip5_trains , cmip5_labels = make_flatted( cmip_train , cmip_label , cmip_month_sin , cmip_month_cos , cmip5_info ,
                                            cmip6_info[ 1 ] * cmip6_info[ 2 ] )

# 得到扁平化后的数据维度为（模式数×序列长度×纬度×经度×特征数），其中序列长度=年数×12
print( np.shape( soda_trains ) , np.shape( cmip6_trains ) , np.shape( cmip5_trains ) )

del soda_month_sin , soda_month_cos
del cmip_month_sin , cmip_month_cos
del soda_train , soda_label
del cmip_train , cmip_label

'''
空值填充
'''
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

'''
构造数据集
我们从CMIP数据的每个模式中各抽取100条数据作为训练集（这里抽取的样本数只是作为一个示例，
实际模型训练的时候使用多少样本需要综合考虑可用的资源条件和构建的模型深度），从SODA模式中抽取100条数据作为验证集。
有的同学可能会疑惑，既然这里只用了100条SODA数据，那么为什么还要对SODA数据扁平化后再抽样而不直接用原始数据呢，
因为直接取原始数据的前12个月作为输入，后24个月作为标签所得到的验证集每一条都是从0月开始的，而线上的测试集起始月份是随机抽取的，
因此这里仍然要尽可能保证验证集与测试集的数据分布一致，使构建的验证集的起始月份也是随机的。
'''
def build_dataset(cmip_trains, cmip_labels, num_models, num_samples):
    X, y = [], []
    for model_i in range(num_models):
        indices = np.random.choice(cmip_trains.shape[1] - 12, size=num_samples)
        for ind in indices:
            X.append(cmip_trains[model_i, ind:ind + 12])
            y.append(cmip_labels[model_i][ind:ind + 24])
    return np.array(X), np.array(y)

# 构造训练集
num_samples = 100
X_train, y_train = build_dataset(cmip5_trains, cmip5_labels, 17, num_samples)
X_tmp, y_tmp = build_dataset(cmip6_trains, cmip6_labels, 15, num_samples)
X_train = np.concatenate((X_train, X_tmp))
y_train = np.concatenate((y_train, y_tmp))

# 构造测试集
X_valid, y_valid = build_dataset(soda_trains, soda_labels, 1, num_samples)


# 查看数据集维度
X_train.shape , y_train.shape , X_valid.shape , y_valid.shape
del cmip5_trains , cmip5_labels
del cmip6_trains , cmip6_labels
del soda_trains , soda_labels

# 保存数据集
np.save( 'X_train_sample.npy' , X_train )
np.save( 'y_train_sample.npy' , y_train )
np.save( 'X_valid_sample.npy' , X_valid )
np.save( 'y_valid_sample.npy' , y_valid )



if __name__ == '__main__' :
    print( "finished!" )
