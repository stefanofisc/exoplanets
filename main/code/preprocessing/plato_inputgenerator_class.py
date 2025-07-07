"""
    Author: Stefano Fiscale
    Date: 2025-07-04
    
    This file generate the input signals from the phase_flux data stored in "join_FittedEvents_AllParameters.ftr".
    The output <global_view, event_id, label> is stored into a csv file in the data/ folder.
"""
#import  numpy                      as np #decomment for horizontal reflection
import  sys
import  matplotlib.pyplot           as plt
import  lightkurve                  as lk
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
        self.__dataset  =   DatasetAllParameters(GlobalPaths.PLATO_RAW_DATA / GlobalPaths.plato_fitted_events_ftr_file) #type: <class 'lam_1_table_allparameters.DatasetAllParameters'>
        self.__log      =   Logger()
        
        self.__dataprocessor = DataProcessor()

        # Data structures to store output data
        self.__global_views = []
        self.__global_views_id = []
        self.__global_views_labels = []
    
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
            #NOTE. Io non credo che EB_dur sia in ore
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
        #plt.plot(lc.time.value, np.flip(lc.flux.value), '.', linewidth=1, label='Horizontal reflection')

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
        output_lc   = self.__dataprocessor.preprocessing_pipeline_from_binning(
            input_lc
        )                
        # Append the data <global_view, event_id, label> in the related data structures.
        self.__global_views.append( output_lc.flux.value )
        self.__global_views_id.append( event_id )
        self.__global_views_labels.append( label )

    def generate_input_records(self, output_csv_file:str):
        """Iterate over the entire input csv file and generates the global views"""
        # Convert labels from categorical to integer
        my_df = self.__dataset._convert_labels_from_categorical_to_int(
            column_name = FittedEventsColumns.LABEL,
            mapping = self.__define_mapping_labels()
        )

        # Iterate over the input csv file
        for idx, tce in tqdm(my_df.iterrows(), total=len(my_df), desc="Processing Records"):
            # Get event id and label.
            event_id    = tce[FittedEventsColumns.EVENT_ID]
            label       = tce[FittedEventsColumns.LABEL]

            try:
                # Extract phase time and phase flux (type: <class 'numpy.ndarray'>, length: 500)
                phase_time = tce[FittedEventsColumns.PHASE_TIME]
                phase_flux = tce[FittedEventsColumns.PHASE_FLUX]
                # Binning flux data
                self.__generate_single_record(phase_time, phase_flux)

            except Exception as e:
                self.__log.error(f'In class Preprocessing, generate_input_records() --> {e}')
        
        self.__dataprocessor.save_output_to_csv(
            output_csv_file     = GlobalPaths.CSV / output_csv_file,
            global_views        = self.__global_views,
            global_views_id     = self.__global_views_id,
            global_views_labels = self.__global_views_labels
            )
        
        del my_df

    def main(self, output_csv_file:str):
        self.generate_input_records(output_csv_file)

    def __del__(self):
        self.__log.info('Distruttore della classe')


if __name__ == '__main__':
    p = InputGenerator()
    p.main('FittedEvents_phase_flux.csv')
