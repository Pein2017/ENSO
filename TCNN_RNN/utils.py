'''
作者：卢沛安
时间：2023年08月31日
'''
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def load_test_data( test_path , test_label_path ) :
    files = os.listdir( test_path )
    X_test = [ ]
    y_test = [ ]
    for file in files :
        X_test.append( np.load( os.path.join( test_path , file ) ) )
        y_test.append( np.load( os.path.join( test_label_path , file ) ) )

    X_test = np.array( X_test )
    y_test = np.array( y_test )

    return X_test , y_test


#
# 构建TCN单元
class TCNBlock( nn.Module ) :
    def __init__( self , in_channels , out_channels , kernel_size , stride , padding ) :
        super().__init__()
        self.bn1 = nn.BatchNorm1d( in_channels )
        self.conv = nn.Conv1d( in_channels , out_channels , kernel_size , stride , padding )
        self.bn2 = nn.BatchNorm1d( out_channels )

        if in_channels == out_channels and stride == 1 :
            self.res = lambda x : x
        else :
            self.res = nn.Conv1d( in_channels , out_channels , kernel_size=1 , stride=stride )

    def forward( self , x ) :
        # 转换输入形状
        N , T , C , H , W = x.shape
        x = x.permute( 0 , 3 , 4 , 2 , 1 ).contiguous()
        x = x.view( N * H * W , C , T )

        # 残差
        res = self.res( x )
        res = self.bn2( res )

        x = F.relu( self.bn1( x ) )
        x = self.conv( x )
        x = self.bn2( x )

        x = x + res

        # 将输出转换回(N,T,C,H,W)的形式
        _ , C_new , T_new = x.shape
        x = x.view( N , H , W , C_new , T_new )
        x = x.permute( 0 , 4 , 3 , 1 , 2 ).contiguous()

        return x


# 构建CNN单元
class CNNBlock( nn.Module ) :
    def __init__( self , in_channels , out_channels , kernel_size , stride , padding ) :
        super().__init__()
        self.bn1 = nn.BatchNorm2d( in_channels )
        self.conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding )
        self.bn2 = nn.BatchNorm2d( out_channels )

        if (in_channels == out_channels) and (stride == 1) :
            self.res = lambda x : x
        else :
            self.res = nn.Conv2d( in_channels , out_channels , kernel_size=1 , stride=stride )

    def forward( self , x ) :
        # 转换输入形状
        N , T , C , H , W = x.shape
        x = x.view( N * T , C , H , W )

        # 残差
        res = self.res( x )
        res = self.bn2( res )

        x = F.relu( self.bn1( x ) )
        x = self.conv( x )
        x = self.bn2( x )

        x = x + res

        # 将输出转换回(N,T,C,H,W)的形式
        _ , C_new , H_new , W_new = x.shape
        x = x.view( N , T , C_new , H_new , W_new )

        return x


class TCNNBlock( nn.Module ) :
    def __init__( self , in_channels , out_channels , kernel_size , stride_tcn , stride_cnn , padding ) :
        super().__init__()
        self.tcn = TCNBlock( in_channels , out_channels , kernel_size , stride_tcn , padding )
        self.cnn = CNNBlock( out_channels , out_channels , kernel_size , stride_cnn , padding )

    def forward( self , x ) :
        x = self.tcn( x )
        x = self.cnn( x )
        return x

# 构造模型


class Model( nn.Module ) :
    def __init__( self ) :
        super().__init__()
        self.conv = nn.Conv2d( 4 , 64 , kernel_size=7 , stride=2 , padding=3 )
        self.tcnn1 = TCNNBlock( 64 , 64 , 3 , 1 , 1 , 1 )
        self.tcnn2 = TCNNBlock( 64 , 128 , 3 , 1 , 2 , 1 )
        self.tcnn3 = TCNNBlock( 128 , 128 , 3 , 1 , 1 , 1 )
        self.tcnn4 = TCNNBlock( 128 , 256 , 3 , 1 , 2 , 1 )
        self.tcnn5 = TCNNBlock( 256 , 256 , 3 , 1 , 1 , 1 )
        self.rnn = nn.RNN( 256 , 256 , batch_first=True )
        self.maxpool = nn.MaxPool1d( 2 )
        self.fc = nn.Linear( 256 * 3 , 24 )

    def forward( self , x ) :
        # 转换输入形状
        N , T , H , W , C = x.shape
        x = x.permute( 0 , 1 , 4 , 2 , 3 ).contiguous()
        x = x.view( N * T , C , H , W )

        # 经过一个卷积层
        x = self.conv( x )
        _ , C_new , H_new , W_new = x.shape
        x = x.view( N , T , C_new , H_new , W_new )

        # TCNN部分
        for i in range( 3 ) :
            x = self.tcnn1( x )
        x = self.tcnn2( x )
        for i in range( 2 ) :
            x = self.tcnn3( x )
        x = self.tcnn4( x )
        for i in range( 2 ) :
            x = self.tcnn5( x )

        # 全局平均池化
        x = F.adaptive_avg_pool2d( x , (1 , 1) ).squeeze()

        # RNN部分，分别得到长度为T、T/2、T/4三种时间尺度的特征表达，注意转换RNN层输出的格式
        hidden_state = [ ]
        for i in range( 3 ) :
            x , h = self.rnn( x )
            h = h.squeeze()
            hidden_state.append( h )
            x = self.maxpool( x.transpose( 1 , 2 ) ).transpose( 1 , 2 )

        x = torch.cat( hidden_state , dim=1 )
        x = self.fc( x )

        return x

if __name__ == '__main__' :
    print( "finished!" )
