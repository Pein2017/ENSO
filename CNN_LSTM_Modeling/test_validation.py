from numba import jit
from torch.utils.data import DataLoader

from global_utils import *
from utils import *


@jit( nopython=True )
def fill_test_matrix( first_months ) :
    test_month_sin = np.zeros( (104 , 12 , 24 , 72 , 1) )
    test_month_cos = np.zeros( (104 , 12 , 24 , 72 , 1) )
    for y in range( 104 ) :
        for m in range( 12 ) :
            val_sin = np.sin( 2 * np.pi * ((m + first_months[ y ] - 1) % 12) / 12 )
            val_cos = np.cos( 2 * np.pi * ((m + first_months[ y ] - 1) % 12) / 12 )
            test_month_sin[ y , m , : , : ] = val_sin
            test_month_cos[ y , m , : , : ] = val_cos
    return test_month_sin , test_month_cos



def main() :
    setup_plot_style()
    checkpoint_path = r"D:\PyCharm Community Edition 2022.2.2\ENSO\CNN_LSTM_Modeling\CNN_LSTM_model_weights.pth"
    checkpoint = torch.load( checkpoint_path )
    model = Model()
    model.load_state_dict( checkpoint[ 'state_dict' ] )

    test_path = 'D:/PyCharm Community Edition 2022.2.2/ENSO/data_source/round1_test_data/'
    test_label_path = 'D:/PyCharm Community Edition 2022.2.2/ENSO/data_source/round1_test_labels/'

    X_test , y_test , first_months = load_test_data( test_path , test_label_path )

    test_month_sin , test_month_cos = fill_test_matrix( first_months )

    batch_size = 32
    X_test = np.concatenate( [ X_test , test_month_sin , test_month_cos ] , axis=-1 )
    testset = AIEarthDataset( X_test , y_test )
    testloader = DataLoader( testset , batch_size=batch_size , shuffle=False )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preds = evaluate_model( model , testloader , device , batch_size )

    s = score( y_test , preds )
    print( 'Score: {:.3f}'.format( s ) )


if __name__ == '__main__' :
    main()
    print( "finished!" )
