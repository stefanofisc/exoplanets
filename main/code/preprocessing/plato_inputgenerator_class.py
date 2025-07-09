"""
    Author: Stefano Fiscale
    Date: 2025-07-04
    
    This file generate the input signals from the phase_flux data stored in "join_FittedEvents_AllParameters.ftr".
    The output <global_view, event_id, label> is stored into a csv file in the data/ folder.
    Run this code within the conda environment "preprocessing"
"""
import  pandas                      as pd   #NOTE RIMUOVI
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
        self.__dataset          =   DatasetAllParameters(GlobalPaths.PLATO_DATA_TABLES / GlobalPaths.plato_fitted_events_ftr_file) #type: <class 'lam_1_table_allparameters.DatasetAllParameters'> 
        self.__dataprocessor    =   DataProcessor()

        # Data structures to store output data
        self.__global_views         = []
        self.__global_views_id      = []
        self.__global_views_labels  = []

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

    def plot(self, lc=None, signal_type=999, signal_idx=999, signal_label='global view', time=None, flux=None):
        """
            Plot PLATO light curves.
            Input:
                - lc:                   a lk.LightCurve object
                - signal_type (int):    the label of the signal
                - signal_idx (int):     the event-id of the signal
                - signal_label (str):   the label you want to show in the plot
                - time (numpy.ndarray): time values of the signal
                - flux (numpy.ndarray): flux values of the signal
            
            In order to plot, provide the lc object or time and flux values instead.
        """
        plt.figure(figsize=(10,6))

        if lc is not None:
            plt.plot(lc.time.value, lc.flux.value, '.', linewidth=1, label=signal_label)
            #plt.plot(lc.time.value, np.flip(lc.flux.value), '.', linewidth=1, label='Horizontal reflection')   #NOTE. Horiz.refl.
        elif time is not None and flux is not None:
            plt.plot(time, flux, '.', linewidth=1, label=signal_label)
        else:
            self.__log.error('In class InputGenerator.plot(). Missing time-flux data to plot.')
        
        plt.grid(True, alpha=0.2)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(GlobalPaths.PLOT_DEBUG / 'plato_phase-flux_data_zero-median' / f'{signal_idx}_{signal_type}.png', dpi=600)
        plt.close()

    def __filter_out_rows(self, my_df, event_id_to_reject):
        # Create a boolean mask: True for rows to keep, False otherwise
        rows_to_keep    = ~my_df[FittedEventsColumns.EVENT_ID].isin(event_id_to_reject)
        return          my_df[rows_to_keep].copy() # .copy() to avoid SettingWithCopyWarning

    def __generate_single_record(self, phase_time, phase_flux, event_id, label):
        """
            Generate a single input record <global_view, event_id, label>
            This is the structure of each row in the output csv file
        """
        #NOTE Binning, sconsigliato, forse rimuovo
        #NOTE DECOMMENT input_lc    = lk.LightCurve(time = phase_time, flux = phase_flux)
        #NOTE DECOMMENT output_lc   = self.__dataprocessor.preprocessing_pipeline_from_binning(input_lc)                
        
        # Append the data <global_view, event_id, label> in the related data structures.
        #NOTE DECOMMENT self.__global_views.append( output_lc.flux.value )
        self.__global_views.append( phase_flux )
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
        # Filter out inconsistent phase folded light curves
        my_df   = self.__filter_out_rows(my_df, ['00465-0','06772-0','06820-0','00796-0'])

        # Iterate over the input csv file
        for idx, tce in tqdm(my_df.iloc[1228:1300].iterrows(), total=len(my_df), desc="Processing Records"):
            # Get event id and label.
            event_id    = tce[FittedEventsColumns.EVENT_ID]
            label       = tce[FittedEventsColumns.LABEL]

            try:
                # Extract phase time and phase flux (type: <class 'numpy.ndarray'>, length: 500). No NaN in phase_time arrays
                phase_time = tce[FittedEventsColumns.PHASE_TIME]
                phase_flux = tce[FittedEventsColumns.PHASE_FLUX]

                # Interpolate over NaN in phase_flux arrays
                if self.__dataprocessor._check_nan_in_flux(phase_flux) == True:
                    phase_flux  = self.__dataprocessor._interpolate_nan(phase_flux)

                #NOTE Salva questo blocco in metodo in dataprocessor_class
                if len(phase_flux) > 500:
                    phase_flux      = phase_flux[::2]                               # len(phase_flux) = 500
                    phase_time      = phase_time[::2]
                    phase_flux      = self.__dataprocessor._zero_median(phase_flux)
                #else:
                #    phase_flux      = self.__dataprocessor._zero_median(phase_flux)
                #NOTE DEBUG
                
                self.plot(
                    signal_type = label,
                    signal_idx  = event_id,
                    signal_label= 'phase flux zero-median',
                    time        = phase_time,
                    flux        = phase_flux
                    )
                
                self.__generate_single_record(phase_time, phase_flux, event_id, label)

            except Exception as e:
                self.__log.error(f'In class Preprocessing, generate_input_records() --> {e}')
        """
        self.__dataprocessor.save_output_to_csv(
            output_csv_file     = GlobalPaths.CSV / output_csv_file,
            global_views        = self.__global_views,
            global_views_id     = self.__global_views_id,
            global_views_labels = self.__global_views_labels
            )
        """
        del my_df
    
    def save_signals_and_labels_from_dataframe(self, df, samples_outfile='samples.npy', labels_outfile='labels.npy'):
        """
        Salva i segnali (vettori di lunghezza fissa) e le etichette da un DataFrame in due file .npy.

        Args:
            df (pd.DataFrame): DataFrame in cui ogni riga contiene un vettore di lunghezza 500 e l'ultima colonna contiene l'etichetta.
            samples_outfile (str): Nome del file .npy per salvare i vettori di input (features).
            labels_outfile (str): Nome del file .npy per salvare le etichette.
        """

        # Suddivide in segnali e etichette
        samples = df.iloc[:, :-2].values.astype(np.float32)  # tutte le colonne tranne l'ultima
        labels = df.iloc[:, -1].values                       # solo l'ultima colonna
        self.__log.info(f'samples length: {len(samples)}')
        exit(0)
        # Salvataggio su disco
        np.save(GlobalPaths.PLATO_DATA_NUMPY / samples_outfile, samples)
        np.save(GlobalPaths.PLATO_DATA_NUMPY / labels_outfile, labels)

        print(f"[✓] Dati salvati correttamente in:\n  - {samples_outfile}\n  - {labels_outfile}")

    def main(self, output_csv_file:str):
        self.generate_input_records(output_csv_file)
    
    def __del__(self):
        self.__log.info('Distruttore della classe')


if __name__ == '__main__':
    p = InputGenerator()
    p.main('plato_FittedEvents_phaseflux_original_multiclass.csv')
    #csv_filename = ''
    #df = pd.read_csv('/Users/stefanofisc/Desktop/exoplanets/main/data/main_datasets/csv_format/plato_FittedEvents_phaseflux_original_multiclass.csv')
    #p.save_signals_and_labels_from_dataframe(    df)
