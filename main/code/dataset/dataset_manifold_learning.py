import  sys
#import  pandas      as      pd
import  numpy       as      np
from    dataset     import  TensorDataHandler
from    pathlib     import  Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils       import  GlobalPaths

class DatasetManifoldLearning(TensorDataHandler):
    def __init__(self, manifold_learning_dataset_hyperparameters_object = None):
        super().__init__()

        self.__dataset_conf = manifold_learning_dataset_hyperparameters_object
        
        #NOTE. Questa chiamata va riconsiderata in accordo con le note definite in class ManifoldLearning.project_data()
        self.__load_training_data()
        
        super()._print_tensor_shapes()
    
    def __load_training_data(self):
        """
            Carica i numpy.ndarray (x_train, y_train).
        """    
        super().set_x_y_train(
            np.load(GlobalPaths.DATA_NUMPY / self.__dataset_conf.filename_samples),
            np.load(GlobalPaths.DATA_NUMPY / self.__dataset_conf.filename_labels).astype(int)
        )


#NOTE DEBUG, Temporary method
"""
def save_signals_and_labels(df, label_column='signal_type', signal_outfile='plato_FittedEvents_global_view_original_tdepth_features.npy', label_outfile='plato_FittedEvents_global_view_original_tdepth_labels.npy'):
    ###
    #Salva i segnali e le etichette da un DataFrame in due file separati .npy.
    #
    #Args:
    #    phase_flux_i
    #    df (pd.DataFrame): DataFrame contenente i segnali e le etichette.
    #    label_column (str): Nome della colonna contenente le etichette.
    #    signal_outfile (str): Nome del file .npy per salvare i segnali.
    #    label_outfile (str): Nome del file .npy per salvare le etichette.
    ###
    # Estrai i segnali: presuppone che siano nelle prime 201 colonne
    signals = df.iloc[:, :201].values
    labels  = df[label_column].values

    # Salva su disco
    np.save(GlobalPaths.PLATO_RAW_DATA / signal_outfile, signals)
    np.save(GlobalPaths.PLATO_RAW_DATA / label_outfile, labels)

    print(f"[✓] {signal_outfile} e {label_outfile} salvati con successo.")
"""

if __name__ == '__main__':
    print('dataset manifold learning')