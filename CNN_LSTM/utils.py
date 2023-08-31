'''
作者：卢沛安
时间：2023年08月29日
'''

import os

import numpy as np
from torch import nn

# 指定中文字体文件路径
font_path = os.path.join( 'C:\\' , 'Windows' , 'Fonts' , 'msyh.ttc' )


def load_test_data( test_path , test_label_path ) :
    files = os.listdir( test_path )
    X_test = [ ]
    y_test = [ ]
    first_months = [ ]
    for file in files :
        X_test.append( np.load( test_path + file ) )
        y_test.append( np.load( test_label_path + file ) )
        first_months.append( int( file.split( '_' )[ 2 ] ) )

    return np.array( X_test ) , np.array( y_test ) , np.array( first_months )


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
