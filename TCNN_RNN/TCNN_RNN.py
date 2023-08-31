'''
作者：卢沛安
时间：2023年08月30日
'''
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from global_utils import AIEarthDataset , RMSELoss , train_and_validate
from utils import Model


def main() :
    # 定义共享的路径前缀
    data_path_prefix = r'D:\PyCharm Community Edition 2022.2.2\ENSO\TCNN_RNN\np_data'
    # 读取数据集
    X_train = np.load( f'{data_path_prefix}\\X_train_sample.npy' )
    y_train = np.load( f'{data_path_prefix}\\y_train_sample.npy' )
    X_valid = np.load( f'{data_path_prefix}\\X_valid_sample.npy' )
    y_valid = np.load( f'{data_path_prefix}\\y_valid_sample.npy' )

    batch_size = 32
    trainset = AIEarthDataset( X_train , y_train )
    trainloader = DataLoader( trainset , batch_size=batch_size , shuffle=True )

    validset = AIEarthDataset( X_valid , y_valid )
    validloader = DataLoader( validset , batch_size=batch_size , shuffle=True )

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    model = Model().to( device )
    optimizer = optim.Adam( model.parameters() , lr=1e-3 , weight_decay=0.001 )
    criterion = RMSELoss  # Make sure this is callable, e.g., nn.MSELoss() if RMSELoss is a class
    model_weights = './TCNN_RNN__model_weights.pth'
    epochs = 15

    train_and_validate( epochs=epochs , trainloader=trainloader , validloader=validloader , model=model ,
                        optimizer=optimizer , criterion=criterion , model_weights=model_weights ,
                        device=device , y_valid=y_valid , batch_size=batch_size )


if __name__ == '__main__' :
    main()

    print( "finished!" )
