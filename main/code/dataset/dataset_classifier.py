import  numpy               as      np
from    tensor_data_handler import  TensorDataHandler
from    dataset             import  GlobalPaths, log

class DatasetClassifier(TensorDataHandler):
    def __init__(self, classifier_hyperparameters_object):
      super().__init__()

      self.__dataset_conf = classifier_hyperparameters_object._dataset

      if classifier_hyperparameters_object._classifier.mode == 'train':
        self.__load_training_data()
      
      elif classifier_hyperparameters_object._classifier.mode == 'test':
        self.__load_test_data()
      
      else:
        raise ValueError(f'In DatasetClassifier, mode. Got {classifier_hyperparameters_object._classifier.mode}, but expect "train" or "test"')

      super()._print_tensor_shapes()

    def __load_training_data(self):
      """
        Load the numpy.ndarray(x_train, y_train) from features_step2_mlp/.
      """
      super().set_x_y_train(
        np.load(GlobalPaths.FEATURES_STEP2_MLP / self.__dataset_conf.filename_samples),
        np.load(GlobalPaths.FEATURES_STEP2_MLP / self.__dataset_conf.filename_labels).astype(int)
      )
    
    def __load_test_data(self):
      """
        Load the numpy.ndarray(x_test, y_test) from features_step2_mlp/.
      """
      super().set_x_y_test(
        np.load(GlobalPaths.FEATURES_STEP2_MLP / self.__dataset_conf.filename_samples),
        np.load(GlobalPaths.FEATURES_STEP2_MLP / self.__dataset_conf.filename_labels).astype(int)
      )
        
    def __del__(self):
      log.info('\nDestructor called for the class DatasetMLP')