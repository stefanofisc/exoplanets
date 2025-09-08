import  sys
import  yaml
import  torch
import  pandas                  as      pd
import  numpy                   as      np
from    torch.utils.data        import  TensorDataset, DataLoader
from    sklearn.model_selection import  train_test_split
from    sklearn.preprocessing   import  LabelEncoder
from    collections             import  Counter
from    pathlib                 import  Path
from    dataclasses             import  dataclass
from    typing                  import  Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils                   import  GlobalPaths
from    logger                  import  log

class PathConfigDataset:
    # Collection of input variables shared among the modules
    MAIN_DATASET  = GlobalPaths.DATA  / 'main_datasets'
    CSV           = MAIN_DATASET      / 'csv_format'
    SPLIT         = MAIN_DATASET      / 'csv_format_split_80_20'
    TENSORS       = MAIN_DATASET      / 'tensor_format_split_80_20'
    NUMPY         = MAIN_DATASET      / 'numpy_format_split_80_20'

@dataclass
class InputVariablesDatasetCSV:
    # define type of input data
    _dataset_filename:          str
    _initialize_from_scratch:   bool
    _mapping:                   Optional[dict]
    _test_size:                 Optional[float]
    _dataset_save_split_format: Optional[str]
    _load_tensors:              bool
    _catalog_name:              Optional[str]

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _dataset_filename         = config.get('dataset_filename', None),                             
            _initialize_from_scratch  = config.get('initialize_from_scratch', False),
            _mapping                  = config.get('mapping', {}),
            _test_size                = config.get('test_size', 0.2),
            _dataset_save_split_format= config.get('dataset_save_split_format', None),
            _load_tensors             = config.get('load_tensors', False),
            _catalog_name             = config.get('catalog_name', None)
            )

class Dataset:
    def __init__(self, dataframe):
      """
        Dataset class constructor.
        Input:
          - dataframe: pandas DataFrame;
      """
      self._df        = dataframe.copy()              # The entire pandas DataFrame
      self._train_df  = None                          # Training set .csv
      self._test_df   = None                          # Test set .csv 
      self.__X_train  = None                          # Training set: PyTorch tensor
      self.__y_train  = None
      self.__X_test   = None                          # Test set: PyTorch tensor
      self.__y_test   = None

      self.__dataset_hyperparameters_object = self.__init_dataset_hyperparameters()

      self.label_col      = self._df.columns[-1]   # Last column of dataframe must be the TCE label
      self.label_encoder  = None
      self.label_mapping  = None
      # End of basic initialization, common to all modules using this class

      if self.__dataset_hyperparameters_object._initialize_from_scratch == True:
        log.debug("Initializing from scratch...")
        # Questa parte viene eseguita se e solo se l'oggetto Dataset viene istanziato al fine di processare un nuovo pandas DataFrame
        # Mapping the labels (categorical or numerical)
        if self.__dataset_hyperparameters_object._mapping is not None:
          self.__encode_labels(mapping = self.__dataset_hyperparameters_object._mapping)
        else:
          self.__encode_labels()

        # Splitting into training-test sets        
        if self.__dataset_hyperparameters_object._dataset_save_split_format is not None:

          # Initialize train_df and test_df split
          self._train_df, self._test_df = self.__split(test_size = self.__dataset_hyperparameters_object._test_size)

          format = self.__dataset_hyperparameters_object._dataset_save_split_format

          if format == 'csv':
            self.__save_split_as_csv()
          
          elif format == 'tensor':
            self.__save_split_as_tensors(shuffle_train = True)
          
          elif format == 'numpy':
            self.__save_split_as_numpy(shuffle_train = True)
          
          else:
            raise ValueError(f'Got {format} as format. Accepted values: csv, tensor, numpy.')

      elif self.__dataset_hyperparameters_object._load_tensors == True:
        log.debug('Loading tensors...')
        self.__load_tensors(catalog_name = self.__dataset_hyperparameters_object._catalog_name)
      
      else:
        raise ValueError('[!] initialize_from_scratch or load_tensors must be True')
      
      self.__print_tensor_shapes()
      # End constructor

    def __init_dataset_hyperparameters(self):
       return InputVariablesDatasetCSV.get_input_hyperparameters(GlobalPaths.CONFIG / GlobalPaths.config_dataset_csv_file)

    def count_classes(self, dataframe='main'):
        """
          Print the number of elements for each class.
          Input:
            - dataframe: {main, training, test}
          Example of usage:
          >>> # Print the number of elements per class
          >>> dataset_handler = Dataset(df, config)
          >>> dataset_handler.count_classes()
          >>> dataset_handler.count_classes(dataframe='training')
          >>> dataset_handler.count_classes(dataframe='test')
        """
        counts = None
        if dataframe == 'main':
          log.info('\nShowing samples distribution of the entire dataset')
          counts = Counter(self._df[self.label_col])

        elif dataframe == 'training':
          if self.__dataset_hyperparameters_object._initialize_from_scratch:
            log.info('\nShowing samples distribution of training set')
            counts = Counter(self._train_df[self.label_col])
          else:
            self.__print_tensor_shapes()  # se invece stai lavorando con tensori, chiama questo metodo per mostrare num.el. train-test
        
        elif dataframe == 'test':
          if self.__dataset_hyperparameters_object._initialize_from_scratch:
            log.info('\nShowing samples distribution of test set')
            counts = Counter(self._test_df[self.label_col])
        
        else:
          log.info(f'\nError in Dataset.count_classes(). Input values: dataframe. Expected one from [main,training,test], got {dataframe} instead.\n Showing values for the entire dataset.')
          counts = Counter(self._df[self.label_col])
        
        if counts:
          for label, count in counts.items():
              log.info(f"Class '{label}': {count} elements")
          return dict(counts)

    def __encode_labels(self, mapping=None):
        """
          Map labels to integers.
          Input:
            - mapping (dict): Optional, explicit map {label: integer}. Only required for the TESS Tey2023 dataset..
        """
        if mapping:
            # Code executed for tess_tey2023. Converting categorical labels to integer values according to the input mapping
            self._df[self.label_col] = self._df[self.label_col].map(mapping)
            self.label_mapping = mapping
        else:
            # Code executed for kepler_dr24 and kepler_dr25
            self.label_encoder        = LabelEncoder()
            self._df[self.label_col]  = self.label_encoder.fit_transform(self._df[self.label_col])
            self.label_mapping        = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        log.debug(f"Mapping the labels: {self.label_mapping}")

    def __get_training_test_filename(self):
        """
          Automatically define the output filenames for training-test split
        """
        dataset_name   = (self.__dataset_hyperparameters_object._dataset_filename).split('.csv')[0]
        train_filename = f'{dataset_name}_train_split'
        test_filename  = f'{dataset_name}_test_split'
        return train_filename, test_filename

    def __split(self, test_size=0.2, random_state=42):
        """
          Create a balanced train/test split based on labels.
        """
        stratify_labels   = self._df[self.label_col]
        train_df, test_df = train_test_split(self._df, test_size=test_size, random_state=random_state, stratify=stratify_labels)
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)  

    def __save_split_as_csv(self):
        """
          Save the training and test set splits
        """
        train_filename, test_filename   = self.__get_training_test_filename()

        self._train_df.to_csv(PathConfigDataset.SPLIT / f'{train_filename}.csv', index=False)
        self._test_df.to_csv(PathConfigDataset.SPLIT / f'{test_filename}.csv', index=False)

        log.info(f"\nSaved csv splits into:\n- {PathConfigDataset.SPLIT / f'{train_filename}.csv'}\n- {PathConfigDataset.SPLIT / f'{test_filename}.csv'}")

    def __df_to_format(self, df, shuffle=False):
        """
          Convert pandas Dataframe to the format specified into the dataset_save_split_format variable.
          Input:
            - df (pd.Dataframe): train or test Dataframe
            - shuffle (bool): usually set to True for training dataframes  
        """
        if 'plato' in self.__dataset_hyperparameters_object._dataset_filename:
           # ALl PLATO dataset have: <flux><event_id><label>
           X = df.iloc[:, :-2].values.astype('float32')
        else:
          # Kepler and TESS dataset have: <flux><label>
          X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('long')

        if shuffle:
          idx = np.random.permutation(len(X))
          X, y = X[idx], y[idx]

        if self.__dataset_hyperparameters_object._dataset_save_split_format == 'tensor':
          return torch.tensor(X), torch.tensor(y)
        
        elif self.__dataset_hyperparameters_object._dataset_save_split_format == 'numpy':
          return X, y
        
        else:
           raise ValueError(f'In class Dataset.__df_to_format(), got unexpected format: {self.__dataset_hyperparameters_object._dataset_save_split_format}')

    def __save_split_as_tensors(self, shuffle_train=True):
        """
          Save train-test split as PyTorch tensors (.pt).
          Input:
          - shuffle_train: if True, shuffle the training set samples.
        """
        train_filename, test_filename   = self.__get_training_test_filename()

        self.__X_train, self.__y_train  = self.__df_to_format(self._train_df, shuffle=shuffle_train)
        self.__X_test, self.__y_test    = self.__df_to_format(self._test_df, shuffle=False)

        torch.save((self.__X_train, self.__y_train), PathConfigDataset.TENSORS / f'{train_filename}.pt')
        torch.save((self.__X_test, self.__y_test), PathConfigDataset.TENSORS / f'{test_filename}.pt')

        log.info(f"\nSaved tensors into:\n- {PathConfigDataset.TENSORS / f'{train_filename}.pt'}\n- {PathConfigDataset.TENSORS / f'{test_filename}.pt'}")
    
    def __print_tensor_shapes(self):
        """
          Print the dimensions of the training and testing tensors.
        """
        if self.__X_train is not None and self.__X_test is not None:
            log.info('\nShowing train-test tensors shape:')
            log.info(f"X_train: {self.__X_train.shape}, y_train: {self.__y_train.shape}")
            log.info(f"X_test:  {self.__X_test.shape}, y_test:  {self.__y_test.shape}")
        else:
            log.warning("\n[!] Tensors have not been generated yet. You should call this method: save_as_tensors().")

    def __load_tensors(self, catalog_name):
      """
        Loads train-test splits of a given catalog in PyTorch tensor format (.pt)
        Input:
            catalog_name (str): Catalog name. Allowed values:
                'kepler_dr24', 'kepler_dr25', 'tess_tey23', 'plato_flux_original', 'plato_flux_zeromedian'
      """
      catalog_paths = {
          'kepler_dr24': (
              PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_train_split.pt',
              PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_test_split.pt'
          ),
          'kepler_dr25': (
              PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_train_split.pt',
              PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_test_split.pt'
          ),
          'tess_tey23': (
              PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_train_split.pt',
              PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_test_split.pt'
          ),
          'plato_flux_original': (
              PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_original_multiclass_train_split.pt',
              PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_original_multiclass_test_split.pt'
          ),
          'plato_flux_zeromedian': (
              PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_zeromedian_multiclass_train_split.pt',
              PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_zeromedian_multiclass_test_split.pt'
          )
      }

      if catalog_name not in catalog_paths:
          raise ValueError(
              f"[!] In __load_tensors(): catalog_name must be one of {list(catalog_paths.keys())}, got '{catalog_name}' instead."
          )

      train_tensor_path, test_tensor_path = catalog_paths[catalog_name]

      self.__X_train, self.__y_train      = torch.load(train_tensor_path)
      self.__X_test, self.__y_test        = torch.load(test_tensor_path)

      log.info("\n[✓] Tensors loaded successfully.")

    """
    NOTE. Old. C-oriented
    def __load_tensors(self, catalog_name):
      ###
      #  Loads train-test splits of a given catalog in PyTorch tensor format (.pt)
      #  Input:
      #    - catalog_name: 'kepler_dr24', 'kepler_dr25', 'tess_tey23'
      ###
      if catalog_name == 'kepler_dr24':
        train_tensor_path = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr24_multiclass_test_split.pt'

      elif catalog_name == 'kepler_dr25':
        train_tensor_path = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'kepler_q1-q17_dr25_multiclass_test_split.pt'

      elif catalog_name == 'tess_tey23':
        train_tensor_path = PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'tess_tey2023_multiclass_test_split.pt'
      
      elif catalog_name == 'plato_flux_original':
        train_tensor_path = PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_original_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_original_multiclass_test_split.pt'
        
      elif catalog_name == 'plato_flux_zeromedian':
        train_tensor_path = PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_zeromedian_multiclass_train_split.pt'
        test_tensor_path  = PathConfigDataset.TENSORS / 'plato_FittedEvents_phaseflux_zeromedian_multiclass_test_split.pt'
      
      else:
        raise ValueError(f'Error: Class Dataset, in load_tensors().\ncatalog_name must be in [kepler_dr24, kepler_dr25, tess_tey23], got {catalog_name} instead!')

      self.__X_train, self.__y_train  = torch.load(train_tensor_path)
      self.__X_test, self.__y_test    = torch.load(test_tensor_path)

      log.info("\nTensors loaded successfully")
    """

    def __save_split_as_numpy(self, shuffle_train=True):
        """
          Save train-test split as numpy.ndarray (.npy).
          Input:
          - shuffle_train: if True, shuffle the training set samples.
        """
        train_filename, test_filename   = self.__get_training_test_filename()

        self.__X_train, self.__y_train  = self.__df_to_format(self._train_df, shuffle=shuffle_train)
        self.__X_test, self.__y_test    = self.__df_to_format(self._test_df, shuffle=False)

        np.save(PathConfigDataset.NUMPY / f'{train_filename}_features.npy', self.__X_train)
        np.save(PathConfigDataset.NUMPY / f'{train_filename}_labels.npy', self.__y_train)

        np.save(PathConfigDataset.NUMPY / f'{test_filename}_features.npy', self.__X_test)
        np.save(PathConfigDataset.NUMPY / f'{test_filename}_labels.npy', self.__y_test)

        log.info(f"\nSaved numpy arrays into:\n- {PathConfigDataset.NUMPY / f'{train_filename}_*.npy'}\n- {PathConfigDataset.NUMPY / f'{test_filename}_*.npy'}")

    def get_training_test_samples(self):
      return self.__X_train, self.__y_train, self.__X_test, self.__y_test
    
    def get_full_data_loader(self, batch_size):
      """
        Metodo provvisorio che mi permette di caricare direttamente l'intero dataset D_K25 come training set.
        Il metodo viene richiamato in feature_extractor.py, nel costruttore della classe Model
        Data: 2025-09-08
      """
      data = torch.load(
         GlobalPaths.MAIN_DATASETS / 
         "tensor_format" / 
         "kepler_q1-q17_dr25_multiclass_train_test_split.pt"
      )
      X, y = data
      log.debug(f'X_full: {X.shape}; y_full: {y.shape}')
      dataset = TensorDataset(X, y)
      return DataLoader(dataset, batch_size = batch_size, shuffle = False)

    def get_training_data_loader(self, batch_size):
      """
        Creates a dataset by combining inputs and labels. This method is used by the Model class during training to iterate over the samples.
        Output:
          - DataLoader to iterate in batches of size of your choice.
      """
      train_dataset = TensorDataset(self.__X_train, self.__y_train)
      return DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    def get_test_data_loader(self, batch_size):
      # Analoso di get_training_data_loader, con test set.
      test_dataset = TensorDataset(self.__X_test, self.__y_test)
      return DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    def get_training_test_set_length(self, split='train'):
      if split == 'train':
        return len(self.__X_train)
      else:
        return len(self.__X_test)
    
    def get_training_test_set_labels(self, split='train'):
      if split == 'train':
        return self.__y_train
      else:
        return self.__y_test
    
    def get_catalog_name(self):
      return self.__dataset_hyperparameters_object._catalog_name

    def __del__(self):
      log.info('\nDestructor called for the class Dataset.')

if __name__ == "__main__":
   log.info('Dataset class')