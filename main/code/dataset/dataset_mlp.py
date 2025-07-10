import  torch
import  numpy               as      np
from    tensor_data_handler import  TensorDataHandler
from    dataset             import  GlobalPaths, log

class DatasetMLP(TensorDataHandler):
    def __init__(self, mode, dataset_conf, training_conf = None):
      """
        Class that defines the operations to apply to feature vectors before processing them through MLP.
        Input:
          - mode: train or test
          - dataset_conf: Object of the DatasetConfig class in mlp_class.py that contains,
                        filename_samples: Features extracted from the CNN, in data/features_step1_cnn/
                        filename_labels: The 2D representation of these features, obtained with t-SNE. data/features_step2_tsne/
                        filename_dispositions: The labels associated with the feature vectors, in data/features_step1_cnn/
      """
      super().__init__()
      log.info('Constructor of DatasetMLP')
      self.__dataset_conf = dataset_conf
      self.__training_conf = training_conf
      self.__mode = mode

      self.__disposition_array = []   # <class 'torch.Tensor'>

      if mode == 'train':
        # Init data structures for training the MLP
        self.__x_train_numpy = []
        self.__x_train_numpy_norm = []
        self.__y_train_numpy = []       # Store the t-SNE two-dimensional coordinates
        self.__y_train_numpy_norm = []
        # Loading and preprocessing
        self.__load_training_data()
        self.__normalize_data()
        self.__init_training_tensors()        
      else:
        self.__x_test_numpy = []
        self.__x_test_numpy_norm = []
        self.__load_test_data()
        self.__normalize_data(normalize_labels = False)
        self.__init_testset_tensors()
      
      self.__load_dispositions()

      super()._print_tensor_shapes()
      
    
    def __load_training_data(self):
      """
        Load numpy.ndarrays (self.__x_train_numpy, y_train_numpy) from features_step1_cnn/ and features_step2_tsne/, respectively.
        These vectors consist of (a) and (e) in the figure in Section 2025-06-12 of the Google Doc.  
      """
      self.__x_train_numpy = np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__dataset_conf.filename_samples)
      self.__y_train_numpy = np.load(GlobalPaths.FEATURES_STEP2_TSNE / self.__dataset_conf.filename_labels)
    
    def __load_test_data(self):
      """
        Load numpy.ndarray (self.__x_test_numpy) from features_step1_cnn/.
        This vector consists of (c) from the figure in Section 2025-06-12 of the Google Doc.
      """
      self.__x_test_numpy = np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__dataset_conf.filename_samples)

    def __load_dispositions(self):
      """
        Load the layouts associated with the feature vectors. These layouts correspond
        to the files in features_step1_cnn/ named '*_labels.npy'.
        Remember that here x = feature vector, y = feature vector projected into 2D space (from tsne).
      """ 
      self.__disposition_array = np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__dataset_conf.filename_dispositions)

    def __normalize_data(self, normalize_labels = True):
        """
          Normalize data to zero mean and unit variance
        """
        epsilon = 1e-8  # offset to improve numerical stability. This prevents division by zero for features with zero std
        if self.__mode == 'train':
          self.__x_train_numpy_norm = (self.__x_train_numpy - self.__x_train_numpy.mean()) / (self.__x_train_numpy.std() + epsilon )
          
          if self.__y_train_numpy is not None and normalize_labels:
              self.__y_train_numpy_norm = (self.__y_train_numpy - self.__y_train_numpy.mean()) / (self.__y_train_numpy.std() + epsilon)
          else:
              self.__y_train_numpy_norm = self.__y_train_numpy
        else:
          self.__x_test_numpy_norm = (self.__x_test_numpy - self.__x_test_numpy.mean()) / (self.__x_test_numpy.std() + epsilon)
        
    def __init_training_tensors(self):
      """
        Convert normalized data to voltage and initialize (X_train, y_train) of the TensorDataHandler class
      """
      super().set_x_y_train( torch.tensor(self.__x_train_numpy_norm, dtype=torch.float32), torch.tensor(self.__y_train_numpy_norm, dtype=torch.float32) )
    
    def __init_testset_tensors(self):
      """
        Convert the normalized data into a tensor and initialize (_X_test) the TensorDataHandler class
      """
      super().set_x_y_test( torch.tensor(self.__x_test_numpy_norm, dtype=torch.float32) )


    def get_training_data_loader(self):
      return super().get_training_data_loader(
        batch_size = self.__training_conf.batch_size,
        dispositions = self.__disposition_array
        )
    
    def get_test_data_loader(self):
      return super().get_test_data_loader()

    def set_dispositions(self, dispositions):
      self.__disposition_array = dispositions #type dispositions = <class 'torch.Tensor'>

    def get_dispositions(self):
      #print(f'[DEBUGGING] get_dispositions(). type disposition_numpy = {type(self.__disposition_array)}')
      return self.__disposition_array         #type dispositions = <class 'torch.Tensor'>

    def __del__(self):
      log.info('\nDestructor called for the class DatasetMLP')