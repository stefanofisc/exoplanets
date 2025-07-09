from torch.utils.data import TensorDataset, DataLoader
import torch
from utils.logger import log

class TensorDataHandler:
    """
      Class that defines common operations to apply to tensors.
      #NOTE. For future development, have Dataset inherit the methods and attributes of TensorDataset and verify the
      correct integration by running tests on the dataset main and in feature_extractor.py.
    """
    def __init__(self):
      self._X_train = None                     # Training set: PyTorch tensor / numpy.ndarray (classifier.py)
      self._y_train = None
      self._X_test = None                      # Test set: PyTorch tensor / numpy.ndarray (classifier.py)
      self._y_test = None
      print('Constructor of TensorDataHandler')

    def _print_tensor_shapes(self):
      """
        Print the dimensions of the training and testing tensors.
      """
      if self._X_train is not None and self._y_train is not None:
          print('\nShowing train tensors shape:')
          print(f"X_train: {self._X_train.shape}, y_train: {self._y_train.shape}")
      if self._X_test is not None:
        if self._y_test is not None:
          print(f"X_test:  {self._X_test.shape}, y_test:  {self._y_test.shape}")
        else:
          print(f"X_test:  {self._X_test.shape}")

    def get_training_test_samples(self):
      return self._X_train, self._y_train, self._X_test, self._y_test
    
    def get_y_train(self):
      return self._y_train

    def get_training_data_loader(self, batch_size, dispositions = None):
      """
        Creates a dataset by combining inputs and labels. This method is used by the Model class during training to iterate over the samples.
        Output:
          - DataLoader to iterate in batches of your choice.
      """
      if dispositions is not None:
        assert len(dispositions) == len(self._X_train), "[!] dispositions deve avere la stessa lunghezza di X_train"

        # Shuffle coerente su X, y e dispositions
        
        indices = torch.randperm(len(self._X_train))
        x_shuffled = self._X_train[indices]
        y_shuffled = self._y_train[indices]
        dispositions_shuffled = torch.tensor(dispositions, dtype=torch.long)[indices]
        
        train_dataset = TensorDataset(x_shuffled, y_shuffled, dispositions_shuffled)
        
        #train_dataset = TensorDataset(self._X_train, self._y_train, torch.tensor(dispositions, dtype=torch.long))
        return DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
      
      else:
        train_dataset = TensorDataset(self._X_train, self._y_train)
        return DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    def get_test_data_loader_with_labels(self, batch_size = 128):
      # Analogous to get_training_data_loader, with test set.
      assert self._y_test is not None, "[!] y_test is None. Cannot create DataLoader with labels."
      test_dataset = TensorDataset(self._X_test, self._y_test)
      return DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    def get_test_data_loader(self, batch_size = 128):
      # When projecting features from test set with MLP, you don't need self._y_test
      test_dataset = TensorDataset(self._X_test)
      return DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    def get_training_test_set_length(self, split='train'):
      if split == 'train':
        return len(self._X_train)
      else:
        return len(self._X_test)
    
    def get_training_test_set_labels(self, split='train'):
      if split == 'train':
        return self._y_train
      else:
        return self._y_test

    def set_x_y_train(self, x_train, y_train):
      """
        Initialize the X_train and y_train tensors from child classes
      """
      self._X_train = x_train
      self._y_train = y_train
    
    def set_x_y_test(self, x_test, y_test = None):
      self._X_test = x_test
      self._y_test = y_test

    def __del__(self):
      log.info('\nDestructor called for the class TensorDataset')