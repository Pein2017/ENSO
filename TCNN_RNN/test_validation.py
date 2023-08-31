'''
作者：卢沛安
时间：2023年08月31日
'''
from torch.utils.data import DataLoader

from global_utils import *
from utils import *


def main() :
    setup_plot_style()
    checkpoint_path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\TCNN_RNN\TCNN_RNN__model_weights.pth"
    checkpoint = torch.load( checkpoint_path )
    model = Model()
    model.load_state_dict( checkpoint[ 'state_dict' ] )

    test_path = 'D:/PyCharm Community Edition 2022.2.2/ENSO/data_source/round1_test_data/'
    test_label_path = 'D:/PyCharm Community Edition 2022.2.2/ENSO/data_source/round1_test_labels/'

    X_test , y_test = load_test_data( test_path , test_label_path )

    batch_size = 32
    testset = AIEarthDataset( X_test , y_test )
    testloader = DataLoader( testset , batch_size=batch_size , shuffle=False )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preds = evaluate_model( model , testloader , device , batch_size )

    s = score( y_test , preds )
    print( 'Score: {:.3f}'.format( s ) )


if __name__ == '__main__' :
    main()
    print( "finished!" )
