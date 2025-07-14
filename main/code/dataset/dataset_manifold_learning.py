import  numpy               as      np
from    dataset             import  GlobalPaths, log
from    tensor_data_handler import  TensorDataHandler

class DatasetManifoldLearning(TensorDataHandler):
    def __init__(self, manifold_learning_dataset_hyperparameters_object = None):
        super().__init__()

        self.__dataset_conf = manifold_learning_dataset_hyperparameters_object
        
        self.__load_training_data()        
        super()._print_tensor_shapes()
    
    def __load_training_data(self):
        """
            Carica i numpy.ndarray (x_train, y_train), che possono consistere in:
                - Features prodotte dal modulo feature_extractor (dati in features_step1_cnn/)
                - Segnali di input grezzi                        (dati in main_datasets/numpy_format_split_80_20/)
            
            Il modulo di Manifold Learning caricherà sempre e solo i dati di training, 
            in quanto i suoi output rappresenteranno le etichette (y_train) del MLP. 
            Sarà il MLP, invece, a processare i dati di test. 
        """    
        super().set_x_y_train(
            np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__dataset_conf.filename_samples),
            np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__dataset_conf.filename_labels).astype(int)
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
    log.info('dataset manifold learning')