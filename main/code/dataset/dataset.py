import yaml
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
# Libraries for class containing global variables
from pathlib import Path

class PathConfigDataset:
    # Collection of input variables shared among the modules
    BASE    = Path('/home/stefanofiscale/Desktop/exoplanets/main/')
    DATA    = BASE / 'data'
    MAIN_DATASET = DATA / 'main_datasets'
    CSV     = MAIN_DATASET / 'csv_format'
    SPLIT   = MAIN_DATASET / 'csv_format_split_80_20'
    TENSORS = MAIN_DATASET / 'tensor_format_split_80_20'
    # Path to this module so that it can be used in combination with feature_extraction
    BASE_DATASET_MODULE = BASE / 'code' / 'dataset'

class Dataset:
    def __init__(self, dataframe, config):
      """
        Costruttore della classe Dataset.
        Input:
          - dataframe: pandas DataFrame;
          - config: oggetto contenente i dati di input definiti in un file di configurazione .yaml.
        Example of usage:
        # Creating a Dataset object from scratch, with config_dataset.yaml as configuration file.
        >>> def main():
        >>>  # Carica il file di configurazione YAML
        >>>  with open("config_dataset.yaml", "r") as f:
        >>>      config = yaml.safe_load(f)

        >>>  # Carica il dataset CSV
        >>>  df = pd.read_csv(PathConfigDataset.CSV / config["dataset_filename"])

        >>>  # Istanzia la classe Dataset ed esegue tutte le operazioni
        >>>  dataset_handler = Dataset(df, config)
        >>>  dataset_handler.count_classes()
        >>>  dataset_handler.count_classes(dataframe='training')
        >>>  dataset_handler.count_classes(dataframe='test')
        >>>  print("Elaborazione completata.")
      """
      self._df = dataframe.copy()             # The entire pandas DataFrame
      self._train_df = None                   # Training set .csv
      self._test_df = None                    # Test set .csv 
      self.__X_train = None                     # Training set: PyTorch tensor
      self.__y_train = None
      self.__X_test = None                      # Test set: PyTorch tensor
      self.__y_test = None

      self._config = config
      self.label_col = self._df.columns[-1]   # Last column of dataframe must be the TCE label
      self.label_encoder = None
      self.label_mapping = None
      # End of basic initialization, common to all modules using this class

      if config.get('initialize_from_scratch') == True:
        print("Initializing from scratch...")
        # questa parte viene eseguita se e solo se l'oggetto Dataset viene istanziato al fine di processare un nuovo pandas DataFrame
        # Mapping the labels (categorical or numerical)
        if 'mapping' in config:
          self.__encode_labels(mapping = config['mapping'])
        else:
          self.__encode_labels()
        # Splitting into training-test sets
        if config.get('dataset_splitting') == True:
          self._train_df, self._test_df = self.__split(test_size=config.get('test_size', 0.2))
          # Save train-test .csv
        if config.get('dataset_save_split_csv') == True:
          self.__save_split(config['train_df_filename'], config['test_df_filename'])
          # Save train-test .pt
        if config.get('dataset_save_split_tensors') == True:
          self.__save_as_tensors(
            train_tensor_path=config['train_tensor_path'],
            test_tensor_path=config['test_tensor_path'],
            shuffle_train=True
          )
          self.__print_tensor_shapes()
      
      elif config.get('load_tensors') == True:
        print('Loading tensors...')
        self.__load_tensors(catalog_name=config['catalog_name'])
      
      else:
        raise ValueError('[!] initialize_from_scratch or load_tensors must be True')


    def count_classes(self, dataframe='main'):
        """
          Stampa il numero di elementi per ogni classe.
          Input:
            - dataframe: {main, training, test}
          Example of usage:
          >>> # Stampa il numero di elementi per classe
          >>> dataset_handler = Dataset(df, config)
          >>> dataset_handler.count_classes()
          >>> dataset_handler.count_classes(dataframe='training')
          >>> dataset_handler.count_classes(dataframe='test')
        """
        counts = None
        if dataframe == 'main':
          print('\nShowing samples distribution of the entire dataset')
          counts = Counter(self._df[self.label_col])

        elif dataframe == 'training':
          if self._config.get('initialize_from_scratch'):
            print('\nShowing samples distribution of training set')
            counts = Counter(self._train_df[self.label_col])
          else:
            self.__print_tensor_shapes()  # se invece stai lavorando con tensori, chiama questo metodo per mostrare num.el. train-test
        
        elif dataframe == 'test':
          if self._config.get('initialize_from_scratch'):
            print('\nShowing samples distribution of test set')
            counts = Counter(self._test_df[self.label_col])
        
        else:
          print(f'\nError in Dataset.count_classes(). Input values: dataframe. Expected one from [main,training,test], got {dataframe} instead.\n Showing values for the entire dataset.')
          counts = Counter(self._df[self.label_col])
        
        if counts:
          for label, count in counts.items():
              print(f"Classe '{label}': {count} elementi")
          return dict(counts)

    def __encode_labels(self, mapping=None):
        """Mappa le etichette in interi.
        Input:
        - mapping (dict): opzionale, mappa esplicita {etichetta: intero}. Necessario solo per il dataset TESS Tey2023.
        """
        if mapping:
            # Code executed for tess_tey2023. Converting categorical labels to integer values according to the input mapping
            self._df[self.label_col] = self._df[self.label_col].map(mapping)
            self.label_mapping = mapping
        else:
            # Code executed for kepler_dr24 and kepler_dr25
            self.label_encoder = LabelEncoder()
            self._df[self.label_col] = self.label_encoder.fit_transform(self._df[self.label_col])
            self.label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        print("Mapping the labels:", self.label_mapping)

    def __split(self, test_size=0.2, random_state=42):
        """Crea uno split bilanciato train/test basato sulle etichette."""
        stratify_labels = self._df[self.label_col]
        train_df, test_df = train_test_split(self._df, test_size=test_size, random_state=random_state, stratify=stratify_labels)
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)  

    def __save_split(self, train_df_filename, test_df_filename):
        """Salva gli split di training e test set"""
        self._train_df.to_csv(PathConfigDataset.SPLIT / train_df_filename, index=False)
        self._test_df.to_csv(PathConfigDataset.SPLIT / test_df_filename, index=False)

        print(f"\nSaved csv splits into:\n- {PathConfigDataset.SPLIT / train_df_filename}\n- {PathConfigDataset.SPLIT / test_df_filename}")

    def __df_to_tensor(self, df, shuffle=False):
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('long')

        if shuffle:
          idx = np.random.permutation(len(X))
          X, y = X[idx], y[idx]

        return torch.tensor(X), torch.tensor(y)

    def __save_as_tensors(self, train_tensor_path, test_tensor_path, shuffle_train=True):
        """
          Salva train/test set come tensori PyTorch (.pt).
          Parametri:
          - train_tensor_path = filename del 'train_data.pt'
          - shuffle_train: se True, esegue lo shuffle del training set.
        """
        #NOTE. Experimental: print(f'In save_as_tensors:\n - Input:\n -- df_train: {len(self._train_df)}\n -- df_test: {len(self._test_df)}')

        self.__X_train, self.__y_train = self.__df_to_tensor(self._train_df, shuffle=shuffle_train)
        self.__X_test, self.__y_test = self.__df_to_tensor(self._test_df, shuffle=False)

        torch.save((self.__X_train, self.__y_train), PathConfigDataset.TENSORS / train_tensor_path)
        torch.save((self.__X_test, self.__y_test), PathConfigDataset.TENSORS / test_tensor_path)

        print(f"\nSaved tensors into:\n- {PathConfigDataset.TENSORS / train_tensor_path}\n- {PathConfigDataset.TENSORS / test_tensor_path}")
    
    def __print_tensor_shapes(self):
        """Stampa le dimensioni dei tensori di training e test."""
        if hasattr(self, "X_train") and hasattr(self, "X_test"):
            print('\nShowing train-test tensors shape:')
            print(f"X_train: {self.__X_train.shape}, y_train: {self.__y_train.shape}")
            print(f"X_test:  {self.__X_test.shape}, y_test:  {self.__y_test.shape}")
        else:
            print("\n[!] Tensors have not been generated yet. You should call this method: save_as_tensors().")

    def __load_tensors(self, catalog_name):
      """
        Carica train-test split di un dato catalogo in formato tensori PyTorch (.pt)
        Input:
          - catalog_name: 'kepler_dr24', 'kepler_dr25', 'tess_tey23'
      """
      if catalog_name == 'kepler_dr24':
        train_tensor_path = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_test_split.pt'

      elif catalog_name == 'kepler_dr25':
        train_tensor_path = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_test_split.pt'

      elif catalog_name == 'tess_tey23':
        train_tensor_path = PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_test_split.pt'
      
      else:
        raise ValueError(f'Error: Class Dataset, in load_tensors().\ncatalog_name must be in [kepler_dr24, kepler_dr25, tess_tey23], got {catalog_name} instead!')

      self.__X_train, self.__y_train = torch.load(train_tensor_path)
      self.__X_test, self.__y_test = torch.load(test_tensor_path)

      print("\nTensors loaded successfully")

    def get_training_test_samples(self):
      return self.__X_train, self.__y_train, self.__X_test, self.__y_test
    
    def get_training_data_loader(self, batch_size):
      """
        Crea un dataset combinando input e label. Metodo utilizzato dalla classe Model durante il training, per iterare sui campioni.
        Output:
          - DataLoader per iterare in batch di size a tua scelta.
      """
      train_dataset = TensorDataset(self.__X_train, self.__y_train)
      return DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    def get_training_set_length(self):
      return len(self.__X_train)
    
    def get_training_set_labels(self):
      return self.__y_train

    def __del__(self):
      print('\nDestructor called for the class Dataset.')


def main_dataset_class():
    # Carica il file di configurazione YAML
    with open("config_dataset.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Carica il dataset CSV
    df = pd.read_csv(PathConfigDataset.CSV / config["dataset_filename"])

    # Istanzia la classe Dataset ed esegue tutte le operazioni
    dataset_handler = Dataset(df, config)
    dataset_handler.count_classes()
    dataset_handler.count_classes(dataframe='training')
    dataset_handler.count_classes(dataframe='test')

    x_train, y_train, x_test, y_test = dataset_handler.get_training_test_samples()
    print('\n Testing the method for loading the training test samples')
    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {x_test.shape}, y_test:  {y_test.shape}")

    del dataset_handler, df
    

if __name__ == "__main__":
    main_dataset_class()