'''
作者：卢沛安
时间：2023年08月29日
'''
from Modeling.utils import *
import os
# 其它相关导入
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

custom_font = FontProperties( fname=font_path )
plt.rcParams[ 'font.family' ] = custom_font.get_name()


def train_and_validate( epochs , trainloader , validloader , model , optimizer , criterion , model_weights , device ) :
    train_losses , valid_losses = [ ] , [ ]
    scores = [ ]
    best_score = float( '-inf' )
    preds = np.zeros( (len( y_valid ) , 24) )

    # 省略原有的训练和验证逻辑，可以直接复制粘贴
    for epoch in range( epochs ) :
        print( 'Epoch: {}/{}'.format( epoch + 1 , epochs ) )

        # 模型训练
        model.train()
        losses = 0
        for data , labels in tqdm( trainloader ) :
            data = data.to( device )
            labels = labels.to( device )
            optimizer.zero_grad()
            pred = model( data )
            loss = criterion( pred , labels )
            losses += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        train_loss = losses / len( trainloader )
        train_losses.append( train_loss )
        print( 'Training Loss: {:.3f}'.format( train_loss ) )

        # 模型验证
        model.eval()
        losses = 0
        with torch.no_grad() :
            for i , data in tqdm( enumerate( validloader ) ) :
                data , labels = data
                data = data.to( device )
                labels = labels.to( device )
                pred = model( data )
                loss = criterion( pred , labels )
                losses += loss.cpu().detach().numpy()
                preds[ i * batch_size :(i + 1) * batch_size ] = pred.detach().cpu().numpy()
        valid_loss = losses / len( validloader )
        valid_losses.append( valid_loss )
        print( 'Validation Loss: {:.3f}'.format( valid_loss ) )
        s = score( y_valid , preds )
        scores.append( s )
        print( 'Score: {:.3f}'.format( s ) )

        # 保存最佳模型权重
        if s > best_score :
            best_score = s
            checkpoint = {'best_score' : s ,
                          'state_dict' : model.state_dict()}
            torch.save( checkpoint , model_weights )

    training_vis( train_losses , valid_losses )


# 绘制训练/验证曲线
def training_vis( train_losses , valid_losses ) :
    # 绘制损失函数曲线
    fig = plt.figure( figsize=(8 , 4) )
    # subplot loss
    ax1 = fig.add_subplot( 121 )
    ax1.plot( train_losses , label='train_loss' )
    ax1.plot( valid_losses , label='val_loss' )
    ax1.set_xlabel( 'Epochs' )
    ax1.set_ylabel( 'Loss' )
    ax1.set_title( 'Loss on Training and Validation Data' )
    ax1.legend()
    plt.tight_layout()
    plt.show( block=True )


if __name__ == '__main__' :
    # 定义共享的路径前缀
    data_path_prefix = r'D:\PyCharm Community Edition 2022.2.2\ENSO'
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to( device )
    optimizer = optim.Adam( model.parameters() , lr=1e-3 , weight_decay=0.001 )
    criterion = RMSELoss
    model_weights = './CNN_LSTM_model_weights.pth'
    epochs = 10

    train_and_validate( epochs , trainloader , validloader , model , optimizer , criterion , model_weights , device )
    print( "Finished!" )
