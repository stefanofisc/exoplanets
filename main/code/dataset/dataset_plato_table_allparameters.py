"""
    Author: Stefano Fiscale
    Date: 2025-05-26
    
    This file analyzes the content of the file: sim00465_06899_AllParameters.ftr
"""
import  pandas          as      pd
import  numpy           as      np
from    collections     import  Counter
from    pathlib         import  Path
from    dataset         import  GlobalPaths, log

class DatasetAllParameters:
    def __init__(self, filepath):
        """
            Classe per processare tabelle memorizzate in feather file (.ftr)
            Costruttore per inizializzare l'oggetto DatasetAllParameters e carica il DataFrame da file .ftr
        """
        self.__df   = pd.read_feather(filepath)
        self.__log  = log
        
        self.__signal_type_values = []
        self.__init_signal_type_values()
    
    def __init_signal_type_values(self):
        """
            Salva in __signal_type_values[] le differenti etichette che caratterizzano i dati di questo dataset
        """
        counts = Counter(self.__df['signal_type'])
        for label, _ in counts.items():
            self.__signal_type_values.append(label)
        self.__log.info(f'Questo dataset presenta i labels: {self.__signal_type_values}')

    def __convert_label_column(self):
        """
            Converte la colonna 'signal_type' da np.ndarray a string.
            E' stato necessario farlo la prima volta, in data 2025-05-27, dove ho sovrascritto il nuovo dataframe
        """
        self.__df['signal_type'] = self.__df['signal_type'].apply(
            lambda x: str(x[0]) if isinstance(x, np.ndarray) and len(x) > 0 else ''
        )
    
    def _convert_labels_from_categorical_to_int(self, column_name:str, mapping:dict):
        """
            Questo metodo converte le etichette di una specifica colonna da stringhe a interi,
            in accordo con il mapping fornito in input.

            Args:
                column_name (str): Il nome della colonna nel DataFrame i cui valori devono essere convertiti.
                mapping (dict): Un dizionario dove le chiavi sono le etichette categoriche (stringhe)
                                e i valori sono gli interi a cui devono essere mappate.
                                Esempio: {'EBpri': 0, 'Planet': 1, 'cont': 2, ...}

            Returns:
                pd.DataFrame: Il DataFrame con la colonna specificata convertita.
                            Restituisce una copia del DataFrame modificato.
        """
        if column_name not in self.__df.columns:
            raise ValueError(f"Column {column_name} doesn't exist in this DataFrame.")
        
        # Apply mapping
        # .map() is the most efficient method for such operations 
        # If a value in the column is not found in mapping, it will be replaced by NaN
        self.__df[column_name] = self.__df[column_name].map(mapping)

        # Handle here cases in which you have NaN after mapping
        if self.__df[column_name].isnull().any():
            unmapped_values = self.__df[self.__df[column_name].isnull()][column_name].unique()
            if len(unmapped_values) > 0:
                self.__log.warning(f"Found values in column '{column_name}' not present in mapping, thus converted to NaN: {unmapped_values}")
        
        #self.__log.debug(f'Column values after mapping: {self.__df[column_name].unique()}')

        return self.__df 

    def save_dataframe(self, output_filename:str):
        """
            Salva il DataFrame in formato Feather
        """
        self.__df.to_feather(output_filename)
    
    def get_main_info(self):
        """
            Stampa info generali, intestazione e conteggio classi
        """
        print(self.__df.info())
        print(self.__df.head(5))

        counts = Counter(self.__df['signal_type'])
        for label, count in counts.items():
            print(f"Classe '{label}': {count} elements")

        print(f"Number of PLATO Input Catalog targets\n-- {self.__df['PIC'].count()}")

    def column_describe(self, column_name:str):
        """
            Stampa statistiche descrittive per la colonna selezionata
        """
        self.__log.info(f'{column_name} column:\n{self.__df[column_name].describe()}')

    def row_print_values(self, signal_type:str):
        """
            Stampa tutti i valori di una riga in base all'etichetta in input.
            Questo metodo mi serve in fase di analisi dei dati di input.
            Voglio capire se i parametri P_day, t0_day e Tdur sono nulli quando
            signal_type assume un valore diverso da Planet. Viceversa per i parametri
            P_EB, t0_EB, Tdur_EB
        """
        try:
            for _, tce in self.__df.iterrows():
                if tce['signal_type'] == signal_type:
                    for col_name, col_value in tce.items():
                        print(f'   {col_name}: {col_value}')
                    break
        except Exception as e:
            print(f"An error occurred while reading or iterating the feather file: {e}")

    def get_dataframe(self):
        return self.__df
    
    def main(self):
        pass
        #self.row_print_values('EBpri&secP/2 cont')
        #self.get_main_info()
        #self.column_describe('index_sim')
    
    def __del__(self):
        self.__log.info("Distruttore classe DatasetAllParameters")

if __name__ == '__main__':    
    pass
    # Open dataframe
    #df = DatasetAllParameters(GlobalPaths.PLATO_RAW_DATA / GlobalPaths.fitted_events_ftr_file)

    #df.get_main_info()
    
    #del df