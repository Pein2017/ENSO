'''
作者：卢沛安
时间：2023年08月30日
'''
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from tqdm import tqdm

# 指定中文字体文件路径
font_path = os.path.join( 'C:\\' , 'Windows' , 'Fonts' , 'msyh.ttc' )
custom_font = FontProperties( fname=font_path )
plt.rcParams[ 'font.family' ] = custom_font.get_name()



def evaluate_model( model , testloader , device , batch_size=32 ) :
    model.eval()
    model.to( device )
    preds = np.zeros( (len( testloader.dataset ) , 24) )
    for i , data in tqdm( enumerate( testloader ) ) :
        data , labels = data
        data = data.to( device )
        labels = labels.to( device )
        pred = model( data )
        preds[ i * batch_size : (i + 1) * batch_size ] = pred.detach().cpu().numpy()
    return preds




def setup_plot_style() :
    color = sns.color_palette()
    sns.set_style( 'darkgrid' )
    font_path = os.path.join( 'C:' , 'Windows' , 'Fonts' , 'msyh.ttc' )
    custom_font = FontProperties( fname=font_path )
    plt.rcParams[ 'font.family' ] = custom_font.get_name()



def train_one_epoch( trainloader , model , optimizer , criterion , device ) :
    model.train()
    train_loss = 0.0
    for data , labels in tqdm( trainloader ) :
        data , labels = data.to( device ) , labels.to( device )
        optimizer.zero_grad()
        output = model( data )
        loss = criterion( output , labels )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len( trainloader )


def validate_one_epoch( validloader , model , criterion , device , preds , y_valid , batch_size ) :
    model.eval()
    valid_loss = 0.0
    with torch.no_grad() :
        for i , (data , labels) in tqdm( enumerate( validloader ) ) :
            data , labels = data.to( device ) , labels.to( device )
            output = model( data )
            loss = criterion( output , labels )
            valid_loss += loss.item()
            preds[ i * batch_size :(i + 1) * batch_size ] = output.detach().cpu().numpy()
    return valid_loss / len( validloader ) , score( y_valid , preds )


def train_and_validate( epochs , trainloader , validloader , model , optimizer , criterion , model_weights , device ,
                        y_valid , batch_size ) :
    train_losses , valid_losses = [ ] , [ ]
    scores = [ ]
    best_score = float( '-inf' )
    preds = np.zeros( (len( y_valid ) , 24) )

    for epoch in range( epochs ) :
        print( f'Epoch: {epoch + 1}/{epochs}' )
        train_loss = train_one_epoch( trainloader , model , optimizer , criterion , device )
        valid_loss , s = validate_one_epoch( validloader , model , criterion , device , preds , y_valid , batch_size )

        train_losses.append( train_loss )
        valid_losses.append( valid_loss )
        scores.append( s )

        print( f'Training Loss: {train_loss:.3f}' )
        print( f'Validation Loss: {valid_loss:.3f}' )
        print( f'Score: {s:.3f}' )

        if s > best_score :
            best_score = s
            torch.save( {'best_score' : s , 'state_dict' : model.state_dict()} , model_weights )
        # Visualize training and validation losses
    training_vis( train_losses , valid_losses )


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


def seed_everything( seed=42 ) :
    random.seed( seed )
    os.environ[ 'PYTHONHASHSEED' ] = str( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed( seed )
    torch.backends.cudnn.deterministic = True


# 构造数据管道
class AIEarthDataset( Dataset ) :
    def __init__( self , data , label ) :
        self.data = torch.tensor( data , dtype=torch.float32 )
        self.label = torch.tensor( label , dtype=torch.float32 )

    def __len__( self ) :
        return len( self.label )

    def __getitem__( self , idx ) :
        return self.data[ idx ] , self.label[ idx ]


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
