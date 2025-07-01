import sys
import yaml
#import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import GlobalPaths, TrainingMetrics #get_device

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from dataset import DatasetClassifier

#device = get_device()

@dataclass
class ClassifierConfig:
    mode: str
    model: str
    saved_model_name: Optional[str] = None
    # SVM specific parameters
    kernel: Optional[str] = None
    C: Optional[float] = 1.0
    gamma: Optional[str] = None
    decision_function_shape: Optional[str] = None
    # LDA, QDA specific parameters
    solver: Optional[str] = None
    shrinkage: Optional[str] = None # NOTE. As this parameter could assume None, 'auto' or a float value in [0,1], please convert it in float when necessary

@dataclass
class DatasetConfig:
    filename_samples: str
    filename_labels: str

@dataclass
class InputVariablesClassifier:
    _classifier: ClassifierConfig
    _dataset: DatasetConfig

    @classmethod
    def get_input_hyperparameters(cls, filename: str):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        classifier_conf = ClassifierConfig(**config['classifier'])
        dataset_conf = DatasetConfig(**config['dataset'])

        return cls(
            _classifier=classifier_conf,
            _dataset=dataset_conf
        )

class Classifier:
    def __init__(self):
        """Constructor of the class Classifier""" 
        self.__classifier_hyperparameters_object = self.__init_classifier_hyperparameters()
        
        self.__model = self.__init_model_arch()
        print(self.__model)
        #NOTE. Seems support vector machine doesn't run on gpu
        #self.__model.to(device)

        self.__dataset = self.__init_dataset()

        if self.__classifier_hyperparameters_object._classifier.mode == 'train':
            #NOTE. Do something different?
            pass
        
        print('ending constructor classifier')
        pass
    
    def __init_classifier_hyperparameters(self):
        return InputVariablesClassifier.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_classifier.yaml')

    def __init_model_arch(self):
        model = self.__classifier_hyperparameters_object._classifier.model
        if model == 'svm':
            return SVC(
                kernel          = self.__classifier_hyperparameters_object._classifier.kernel, 
                C               = self.__classifier_hyperparameters_object._classifier.C, 
                gamma           = self.__classifier_hyperparameters_object._classifier.gamma, 
                #class_weight    = self.__compute_class_weights(),  #NOTE. Decomment when DatasetClassifier is ready
                decision_function_shape = self.__classifier_hyperparameters_object._classifier.decision_function_shape
            )

        elif model == 'lda':
            return LinearDiscriminantAnalysis(
                solver=self.__classifier_hyperparameters_object._classifier.solver,
                shrinkage=self.__classifier_hyperparameters_object._classifier.shrinkage,
                priors=None,    #NOTE. Consider using priors in future implementations?
                n_components=self.__classifier_hyperparameters_object._classifier.n_components
            )
        
        elif model == 'qda':
            return QuadraticDiscriminantAnalysis()
        
        else:
            raise ValueError(f'In class Classifier, type of model. Got {model}, but models available are: svm, qda, lda.')

    def __compute_class_weights(self):
        pass
        """
        #NOTE. Decomment when DatasetClassifier object is available. You need to access to y_train to compute class weights
        #Calcola i pesi delle classi con Inverse Frequency Method
        unique_classes  = np.unique(self._y_train)
        class_weights   = compute_class_weight('balanced', classes=unique_classes, y=self._y_train.ravel()) # class_weights: numpy.ndarray
        print(f"Classes values: {unique_classes}")
        print(f"Class weights values: {class_weights}")
        return dict(zip(unique_classes, class_weights))
        """

    def __init_dataset(self):
        return DatasetClassifier(
            self.__classifier_hyperparameters_object
        )

    def __train(self):
        pass
        """
        #NOTE. Decomment when DatasetClassifier object is available and allows you to access data.
        #Train the classifier
        try:
            self.__model.fit(self._x_train, self._y_train)
        except Exception as e1:
            print(f'In class Classifier, __train(). Error: {e1}')
        
        print("Training completed")
        """
    
    def __evaluate(self):
        """
        #NOTE. Decomment this method when DatasetClassifier is available
        # Model assessment on test set
        self.__plot_decision_boundary()
        y_pred = self.__model.predict(self._x_test)
        self._metric_tracker.compute_evaluation_metrics(y_pred, y_pred, self._y_test)   #NOTE. To be changed in TrainingMetrics. Ask to GPT
        """
    
    def __plot_decision_boundary(self):
        pass

if __name__ == '__main__':
    print('classifier module')
    classifier = Classifier()