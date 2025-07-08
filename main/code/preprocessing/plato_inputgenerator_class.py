"""
    Author: Stefano Fiscale
    Date: 2025-07-04
    
    This file generate the input signals from the phase_flux data stored in "join_FittedEvents_AllParameters.ftr".
    The output <global_view, event_id, label> is stored into a csv file in the data/ folder.
    Run this code within the conda environment "preprocessing"
"""
import  sys
import  matplotlib.pyplot           as      plt
import  lightkurve                  as      lk
import  numpy                       as      np 
from    tqdm                        import  tqdm
from    pathlib                     import  Path
from    dataprocessor_class         import  DataProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils                       import  GlobalPaths
from    logger                      import  Logger
from    columns_config              import  FittedEventsColumns
from    labels_config               import  SignalType

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from    dataset_plato_table_allparameters   import DatasetAllParameters

class InputGenerator:
    def __init__(self):
        self.__log              =   Logger()
        self.__dataset          =   DatasetAllParameters(GlobalPaths.PLATO_RAW_DATA / GlobalPaths.plato_fitted_events_ftr_file) #type: <class 'lam_1_table_allparameters.DatasetAllParameters'>
        self.__dataprocessor    =   DataProcessor()

        # Data structures to store output data
        self.__global_views         = []
        self.__global_views_id      = []
        self.__global_views_labels  = []
    
    def __check_nan_in_flux(self, np_array):
        """
            Verifica se un singolo numpy.ndarray contiene almeno un NaN.

            Args:
                np_array (numpy.ndarray): L'array NumPy da controllare.

            Returns:
                bool: True se l'array contiene almeno un NaN, False altrimenti.
        """
        # np.isnan(np_array) crea un array booleano dove True indica un NaN
        # np.any() controlla se c'è almeno un True nell'array booleano
        return np.any(np.isnan(np_array))

    def __define_mapping_labels(self):
        """
            Genera dinamicamente il mapping da etichette stringa a interi
            basandosi sulla classe SignalType.
        """
        mapping = {}
        # Assegna 0 a tutti i tipi di ECLIPSING_BINARIES
        for label in SignalType.ECLIPSING_BINARIES:
            mapping[label] = 0
        
        # Assegna 1 a tutti i tipi di PLANET
        for label in SignalType.PLANET:
            mapping[label] = 1
            
        # Assegna 2 a tutti i tipi di CONTAMINANT
        for label in SignalType.CONTAMINANT:
            mapping[label] = 2
            
        return mapping
    
    def __get_orbital_features(self, label):
        """
            #NOTE. Questo metodo servirà quando dovrai lanciare il metodo DataProcessor.preprocessing_pipeline()
            #NOTE. EB_dur è espresso in secondi
            Because orbital features are stored into different columns, depending
            on the type of label, this method determines the correct columns
            from which retrieve the information
        """
        if label in SignalType.ECLIPSING_BINARIES:
            pass
            #return (period, epoch, duration)
        elif label in SignalType.PLANET:
            pass
        elif label in SignalType.CONTAMINANT:
            pass
        else:
            raise ValueError(f"label {label} doesn't exist in this table")

    def plot(self, lc, signal_type, idx):
        #time = np.linspace(0,201,201)
        plt.figure(figsize=(10,6))

        plt.plot(lc.time.value, lc.flux.value, '.', linewidth=1, label='Global view')
        #plt.plot(lc.time.value, np.flip(lc.flux.value), '.', linewidth=1, label='Horizontal reflection')   #NOTE. Horiz.refl.

        plt.grid(True, alpha=0.2)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(GlobalPaths.OUTPUT_FILES / 'plot_plato_lc' / f'{idx}_{signal_type}.png', dpi=1200)
        plt.close()

    def __generate_single_record(self, phase_time, phase_flux):
        """
            Generate a single input record <global_view, event_id, label>
            This is the structure of each row in the output csv file
        """
        input_lc    = lk.LightCurve(time = phase_time, flux = phase_flux)
        output_lc   = self.__dataprocessor.preprocessing_pipeline_from_binning(input_lc)                
        
        # Append the data <global_view, event_id, label> in the related data structures.
        self.__global_views.append( output_lc.flux.value )
        self.__global_views_id.append( event_id )
        self.__global_views_labels.append( label )

    def generate_input_records(self, output_csv_file:str):
        """
            Iterate over the entire input csv file and generates the global views
        """
        # Convert labels from categorical to integer
        my_df = self.__dataset._convert_labels_from_categorical_to_int(
            column_name = FittedEventsColumns.LABEL,
            mapping     = self.__define_mapping_labels()
        )
        cnt_nan = 0
        # Iterate over the input csv file
        for idx, tce in tqdm(my_df.iterrows(), total=len(my_df), desc="Processing Records"):
            # Get event id and label.
            event_id    = tce[FittedEventsColumns.EVENT_ID]
            label       = tce[FittedEventsColumns.LABEL]

            try:
                # Extract phase time and phase flux (type: <class 'numpy.ndarray'>, length: 500)
                #[NOTE DECOMMENT] phase_time = tce[FittedEventsColumns.PHASE_TIME]
                phase_flux = tce[FittedEventsColumns.PHASE_FLUX]
                if self.__check_nan_in_flux(phase_flux) == True:
                    cnt_nan += 1
                #[NOTE DEBUG] Check for nan in phase_flux

                # Binning flux data
                #[NOTE DECOMMENT] self.__generate_single_record(phase_time, phase_flux)

            except Exception as e:
                self.__log.error(f'In class Preprocessing, generate_input_records() --> {e}')
        self.__log.info(f'Signals with at least one NaN: {cnt_nan}')
        """ [NOTE DECOMMENT]
        self.__dataprocessor.save_output_to_csv(
            output_csv_file     = GlobalPaths.CSV / output_csv_file,
            global_views        = self.__global_views,
            global_views_id     = self.__global_views_id,
            global_views_labels = self.__global_views_labels
            )
        """
        del my_df

    def main(self, output_csv_file:str):
        self.generate_input_records(output_csv_file)

    def __del__(self):
        self.__log.info('Distruttore della classe')


if __name__ == '__main__':
    p = InputGenerator()
    p.main('FittedEvents_phase_flux.csv')
