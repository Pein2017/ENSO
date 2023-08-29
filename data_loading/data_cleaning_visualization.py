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

import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
import netCDF4
import random

path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\数据源\round1_train_data_label"
data = netCDF4.Dataset( path + '\SODA_train.nc' )
label = netCDF4.Dataset( path + '\SODA_label.nc' )
label = np.array( label.variables[ 'nino' ] )

print( data.variables[ 'sst' ].shape , label.shape )

# 以SST特征为例，进行海陆掩膜和插值分析

'''
在给定数据中，经度和纬度坐标都是离散的，每隔5度有一个坐标点，在这样的经纬度坐标下的SST值也是离散的，
因此我们以样本0第0月的SST数据为例，用插值函数来拟合经纬度坐标与SST值之间的函数关系，得到平滑的SST分布。
'''
lon = np.array( data.variables[ 'lon' ] )
lat = np.array( data.variables[ 'lat' ] )
x = lon
y = lat
# 以纬度和经度生成网格点坐标矩阵
xx , yy = np.meshgrid( x , y )
# 取样本0第0月的SST值
z = data.variables[ 'sst' ][ 0 , 0 ]
# 采用三次多项式插值，得到z = f(x, y)的函数f
f = interpolate.interp2d( x , y , z , kind='cubic' )
# 判断每个经纬度坐标点是否在陆地上，用空值遮盖陆地部分
lon_grid , lat_grid = np.meshgrid( x - 180 , y )
# 判断坐标矩阵上的网格点是否为陆地
is_on_land = globe.is_land( lat_grid , lon_grid )
is_on_land = np.concatenate( [ is_on_land[ : , x >= 180 ] , is_on_land[ : , x < 180 ] ] , axis=1 )
# 进行陆地掩膜，将陆地的SST设为空值
z[ is_on_land ] = np.nan
# 可视化掩膜结果，黄色为陆地，紫色为海洋
plt.imshow( is_on_land[ : :-1 , : ] )
plt.show( block=True )

plt.imshow( z[ : :-1 , : ] , cmap=plt.cm.RdBu_r )
plt.show( block=True )

# 设置间隔为1°的经纬度坐标网格，用插值函数得到该坐标网格点的SST值
xnew = np.arange( 0 , 356 , 1 )
ynew = np.arange( -65 , 66 , 1 )
znew = f( xnew , ynew )
lon_grid , lat_grid = np.meshgrid( xnew - 180 , ynew )
is_on_land = globe.is_land( lat_grid , lon_grid )
is_on_land = np.concatenate( [ is_on_land[ : , xnew >= 180 ] , is_on_land[ : , xnew < 180 ] ] , axis=1 )
# 同样进行陆地掩膜
znew[ is_on_land ] = np.nan
# 绘制平滑后的SST分布图
plt.imshow( znew[ : :-1 , : ] , cmap=plt.cm.RdBu_r )
plt.show( block=True )

# 绘制0年36个月的海陆掩膜
for i in range( 1 ) :
    plt.figure( figsize=(15 , 18) )
    for j in range( 36 ) :
        x = lon
        y = lat
        xx , yy = np.meshgrid( x , y )
        z = data.variables[ 'sst' ][ i , j ]
        f = interpolate.interp2d( x , y , z , kind='cubic' )

        xnew = np.arange( 0 , 356 , 1 )
        ynew = np.arange( -65 , 66 , 1 )
        znew = f( xnew , ynew )

        lon_grid , lat_grid = np.meshgrid( xnew - 180 , ynew )
        is_on_land = globe.is_land( lat_grid , lon_grid )
        is_on_land = np.concatenate( [ is_on_land[ : , xnew >= 180 ] , is_on_land[ : , xnew < 180 ] ] , axis=1 )
        znew[ is_on_land ] = np.nan
        plt.subplot( 9 , 4 , j + 1 )
        plt.imshow( znew[ : :-1 , : ] , cmap=plt.cm.RdBu_r )
        plt.title( 'sst - year:{}, month:{}'.format( i + 1 , j + 1 ) )
    plt.show( block=True )

'''
可以看到，SST在每12个月中的前4个月和后4个月都较低，在中间4个月时较高，
这说明，海表温度在春季和冬季较低，在夏季和秋季呈现逐渐升高到最高点然后逐渐降低的变化趋势，这与我们的认知常识是相符的。
'''

'''
以CMIP数据集为例，进行缺失值分析
'''
data = netCDF4.Dataset( path + '\CMIP_train.nc' )
label = netCDF4.Dataset( path + '\CMIP_label.nc' )
label = np.array( label.variables[ 'nino' ] )
# 获得陆地的掩膜
lon_grid , lat_grid = np.meshgrid( x - 180 , y )
is_on_land = globe.is_land( lat_grid , lon_grid )
is_on_land = np.concatenate( [ is_on_land[ : , x >= 180 ] , is_on_land[ : , x < 180 ] ] , axis=1 )
mask = np.zeros( data.variables[ 'sst' ].shape , dtype=int )
mask[ : , : , : , : ] = is_on_land[ np.newaxis , np.newaxis , : , : ]

# 查看SST特征的缺失值数量
name = 'sst'
data_ = np.array( data.variables[ name ] )
before_nan = np.sum( np.isnan( data_ ) )
print( 'before:' , before_nan )
# 查看T300特征的缺失值数量
name = 't300'
data_ = np.array( data.variables[ name ] )
before_nan = np.sum( np.isnan( data_ ) )
print( 'before:' , before_nan )

# 查看Va特征的缺失值数量
name = 'va'
data_ = np.array( data.variables[ name ] )
before_nan = np.sum( np.isnan( data_ ) )
print( 'before:' , before_nan )

# 查看Ua特征的缺失值数量
name = 'ua'
data_ = np.array( data.variables[ name ] )
before_nan = np.sum( np.isnan( data_ ) )
print( 'before:' , before_nan )

# 四个气象特征中，SST特征不存在缺失值，Va和Ua特征中的缺失值数量最多。
#
# 统计每年每月中Ua特征的缺失值数量
m = np.zeros( data_.shape[ 0 :2 ] )
for i in range( data_.shape[ 0 ] ) :
    for j in range( data_.shape[ 1 ] ) :
        if np.sum( np.isnan( data_[ i ][ j ] ) ) != 0 :
            m[ i , j ] = np.sum( np.isnan( data_[ i , j ] ) )

# 计算每一年的缺失值
before = np.sum( m , axis=1 )
# 可视化每一年的缺失值数量
plt.plot( before , 'k' )
plt.ylabel( 'nan count' )
plt.xlabel( 'year' )
plt.show( block=True )

#
# 可视化样本1900中Ua特征的分布
plt.imshow( data_[ 1900 , 0 ][ : :-1 , : ] )
plt.show( block=True )
# 上图中白色部分即为缺失值，可以看到，缺失值多数分布在陆地上，我们将陆地部分进行填充，观察填充后Ua的分布。
# 将陆地位置填0
data_[ mask == 1 ] = 0
# 对陆地部分进行填充后缺失值数量大大减少。
# 统计填充后缺失值的数量
after_nan = np.sum( np.isnan( data_ ) )
print( f'before: {before_nan} \nafter: {after_nan} \npercentage: {1 - float( after_nan ) / before_nan:.3f}' )
m = np.zeros( data_.shape[ 0 : 2 ] )
for i in range( data_.shape[ 0 ] ) :
    for j in range( data_.shape[ 1 ] ) :
        if np.sum( np.isnan( data_[ i , j ] ) ) != 0 :
            m[ i , j ] = np.sum( np.isnan( data_[ i , j ] ) )
after = np.sum( m , axis=1 )
# 对比填充前后每一年缺失值的数量
plt.plot( before , 'k' )
plt.plot( after , 'r' )
plt.legend( [ 'before' , 'after' ] )
plt.title( name )
plt.ylabel( 'nan count' )
plt.xlabel( 'year' )
plt.show( block=True )

'''
温度场和风场可视化 
温度高的地方上方的空气就会向温度低的地方流动，形成风。
因此在分析气候问题时，我们往往会把温度和风向放在一起进行可视化。 
'''
# 对温度场SST进行插值，得到插值函数
x = lon
y = lat
xx , yy = np.meshgrid( x , y )
z = data.variables[ 'sst' ][ 0 , 0 ]
f = interpolate.interp2d( x , y , z , kind='cubic' )
# 获得陆地掩膜
lon_grid , lat_grid = np.meshgrid( x - 180 , y )
is_on_land = globe.is_land( lat_grid , lon_grid )
is_on_land = np.concatenate( [ is_on_land[ : , x >= 180 ] , is_on_land[ : , x < 180 ] ] , axis=1 )

# 对Ua和Va进行陆地掩膜
ua = data.variables[ 'ua' ][ 0 , 0 ]
ua[ is_on_land ] = np.nan
va = data.variables[ 'va' ][ 0 , 0 ]
va[ is_on_land ] = np.nan
# 插值后生成平滑的SST分布
xnew = np.arange( 0 , 356 , 1 )
ynew = np.arange( -65 , 66 , 1 )
znew = f( xnew , ynew )

# 对平滑后的SST进行陆地掩膜
lon_grid , lat_grid = np.meshgrid( xnew - 180 , ynew )
is_on_land = globe.is_land( lat_grid , lon_grid )
is_on_land = np.concatenate( [ is_on_land[ : , xnew >= 180 ] , is_on_land[ : , xnew < 180 ] ] , axis=1 )
znew[ is_on_land ] = np.nan

# 绘制温度场
plt.figure( figsize=(15 , 10) )
plt.imshow( znew[ : :-1 , : ] , cmap=plt.cm.RdBu_r )
plt.colorbar( orientation='horizontal' )  # 显示水平颜色条
# 绘制风向场，其实这里准确来说绘制的是风向异常的向量，而非实际的风向
plt.quiver( lon , lat + 65 , ua[ : :-1 , : ] , va[ : :-1 , : ] ,
            alpha=0.8 )  # 在坐标(lon, lat)处绘制与sqrt(ua^2, va^2)成正比长度的箭头
plt.title( 'year0-month0: SST/UA-VA' )
plt.show( block=True )
'''
从上图中可以看出，温度异常SST在0值附近时没有明显的风向异常，而在其他区域风向异常通常由SST值大的地方指向SST值小的地方。
ENSO现象是指在温度场上赤道东太平洋温度持续异常增暖，在风向场上热带东太平洋与热带西太平洋气压变化（表现为风向）相反的现象。
在上图这个样本中没有出现ENSO现象，大家可以用同样的方法绘制并观察存在ENSO现象（Nino3.4指数连续5个月超过0.5℃）的样本的温度和风场。 
'''

if __name__ == '__main__' :
    print( "finished!" )
