'''
作者：卢沛安
时间：2023年08月29日
'''

import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

# 指定中文字体文件路径
font_path = os.path.join( 'C:\\' , 'Windows' , 'Fonts' , 'msyh.ttc' )


# 构造数据管道
class AIEarthDataset( Dataset ) :
    def __init__( self , data , label ) :
        self.data = torch.tensor( data , dtype=torch.float32 )
        self.label = torch.tensor( label , dtype=torch.float32 )

    def __len__( self ) :
        return len( self.label )

    def __getitem__( self , idx ) :
        return self.data[ idx ] , self.label[ idx ]


class Model( nn.Module ) :
    def __init__( self ) :
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d( 6 , 32 , kernel_size=7 , stride=2 , padding=3 ) ,
            nn.BatchNorm2d( 32 ) ,
            nn.ReLU() ,
            nn.Conv2d( 32 , 32 , kernel_size=3 , stride=1 , padding=1 ) ,
            nn.BatchNorm2d( 32 ) ,
            nn.ReLU() ,
            nn.AvgPool2d( kernel_size=2 , stride=2 )
        )
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM( 3456 , 2048 , batch_first=True )
        self.lstm2 = nn.LSTM( 2048 , 1024 , batch_first=True )
        self.fc = nn.Linear( 1024 , 24 )

    def forward( self , x ) :
        N , T , H , W , C = x.shape
        x = x.permute( 0 , 1 , 4 , 2 , 3 ).contiguous()
        x = x.view( N * T , C , H , W )

        # CNN部分
        x = self.cnn( x )

        x = self.flatten( x )

        _ , C_new = x.shape
        x = x.view( N , T , C_new )

        # LSTM部分
        x , _ = self.lstm1( x )
        x , _ = self.lstm2( x )

        x = self.fc( x[ : , -1 , : ] )  # 使用最后一个时间步的输出

        return x


def rmse( y_true , y_preds ) :
    return np.sqrt( mean_squared_error( y_pred=y_preds , y_true=y_true ) )


# 评估函数
def score( y_true , y_preds ) :
    # 相关性技巧评分
    accskill_score = 0
    # RMSE
    rmse_scores = 0
    a = [ 1.5 ] * 4 + [ 2 ] * 7 + [ 3 ] * 7 + [ 4 ] * 6
    y_true_mean = np.mean( y_true , axis=0 )
    y_pred_mean = np.mean( y_preds , axis=0 )
    for i in range( 24 ) :
        fenzi = np.sum( (y_true[ : , i ] - y_true_mean[ i ]) * (y_preds[ : , i ] - y_pred_mean[ i ]) )
        fenmu = np.sqrt(
            np.sum( (y_true[ : , i ] - y_true_mean[ i ]) ** 2 ) * np.sum( (y_preds[ : , i ] - y_pred_mean[ i ]) ** 2 ) )
        cor_i = fenzi / fenmu
        accskill_score += a[ i ] * np.log( i + 1 ) * cor_i
        rmse_score = rmse( y_true[ : , i ] , y_preds[ : , i ] )
        rmse_scores += rmse_score
    return 2 / 3.0 * accskill_score - rmse_scores


def RMSELoss( y_pred , y_true ) :
    loss = torch.sqrt( torch.mean( (y_pred - y_true) ** 2 , dim=0 ) ).sum()
    return loss
