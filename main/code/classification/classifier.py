import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load

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
    n_components: Optional[int] = 2
    
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

        self.__dataset = self.__init_dataset()

        if self.__classifier_hyperparameters_object._classifier.mode == 'train':
            # Init model arch when in training mode. 
            # Test mode: you don't need this initialization as the model will be loaded in self.__load_model().
            #            Otherwise, this istance will be overwrited and it is just a waste of computational time.
            self.__model = self.__init_model_arch()
            print(self.__model)

        self.__training_metrics = TrainingMetrics()

        #NOTE. Seems support vector machine doesn't run on gpu
        #self.__model.to(device)
        
    
    def __init_classifier_hyperparameters(self):
        return InputVariablesClassifier.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_classifier.yaml')

    def __init_model_arch(self):
        model = self.__classifier_hyperparameters_object._classifier.model
        if model == 'svm':
            return SVC(
                kernel          = self.__classifier_hyperparameters_object._classifier.kernel, 
                C               = self.__classifier_hyperparameters_object._classifier.C, 
                gamma           = self.__classifier_hyperparameters_object._classifier.gamma, 
                class_weight    = self.__compute_class_weights(),
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
        #Calcola i pesi delle classi con Inverse Frequency Method
        unique_classes  = np.unique(self.__dataset.get_y_train())
        class_weights   = compute_class_weight('balanced', classes=unique_classes, y=self.__dataset.get_y_train().ravel()) # class_weights: numpy.ndarray
        print(f"Classes values: {unique_classes}")
        print(f"Class weights values: {class_weights}")
        return dict(zip(unique_classes, class_weights))

    def __init_dataset(self):
        return DatasetClassifier(
            self.__classifier_hyperparameters_object
        )

    def __train(self):
        """Train the classifier"""
        x_train, y_train, _, _ = self.__dataset.get_training_test_samples()
        try:
            self.__model.fit(x_train, y_train)
        except Exception as e1:
            print(f'In class Classifier, __train(). Error: {e1}')
        
        print("Training completed")
    
    def __save_model(self, filename:str):
        """Save the trained model"""
        filepath_base = (
            GlobalPaths.TRAINED_MODELS /
            f'{filename}.joblib'
        )
        if os.path.exists(filepath_base):
            print(f'[WARNING] Filename: {filename}, already exists. \nThe model has not been saved to avoid overwriting.')
        else:
            dump(self.__model, filepath_base)
            print(f'[✓] Model saved in {filepath_base}')

    def __load_model(self, filename:str):
        """
            Load the trained model.
            Input:
                - filename: filename of the model. You need to specify the extension .joblib, but not the full path.
                            The full path will be automatically determined, as it is stored into GlobalPaths.TRAINED_MODELS
        """
        filepath_base = (
            GlobalPaths.TRAINED_MODELS / 
            f'{filename}'
        )
        print(f'[DEBUGGINB] Loading model from: {filepath_base}')
        return load(filepath_base)

    def __evaluate(self):
        """Model assessment"""
        _, _, x_test, y_test = self.__dataset.get_training_test_samples()

        y_pred = self.__model.predict(x_test)
        y_proba = self.__model.predict_proba(x_test) if hasattr(self.__model, "predict_proba") else None

        self.__training_metrics.compute_and_log_classification_metrics(
            y_true = y_test,
            y_pred = y_pred,
            y_proba = y_proba,
            epoch = 0,
            loss = 0.0,
            model_supports_proba = hasattr(self.__model, "predict_proba")
        )

        self.__plot_decision_boundary(output_plot = self.__define_output_plot_filename())

    def __plot_decision_boundary(self, output_plot):
        _, _, x_test, y_test = self.__dataset.get_training_test_samples()
        
        plot_method="pcolormesh" #contour

        ax = plt.gca()

        wdisp = DecisionBoundaryDisplay.from_estimator(
            self.__model,
            x_test,
            plot_method=plot_method,
            #colors="r",
            #levels=[0],
            #alpha=0.5,
            #linestyles=["-"],
            shading="auto",
            response_method="predict",
            ax=ax,
        )
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors="k")
        plt.savefig(output_plot.with_name(output_plot.name), dpi=1200)
        plt.close()

    def __define_model_filename(self):
        model = self.__classifier_hyperparameters_object._classifier.model
        prefix = str(self.__classifier_hyperparameters_object._dataset.filename_samples).split('_train')[0]
        
        if model == 'svm':
            kernel = self.__classifier_hyperparameters_object._classifier.kernel
            filename = f'{prefix}_{model}_{kernel}'
        
        elif model == 'lda' or model == 'qda':
            solver = self.__classifier_hyperparameters_object._classifier.solver
            filename = f'{prefix}_{model}_{solver}'
        
        else:
            raise ValueError(f'In defining output plot filename. Got {model}, when expected svm, lda or qda')

        return filename

    def __define_output_plot_filename(self):
        model = self.__classifier_hyperparameters_object._classifier.model
        prefix = str(self.__classifier_hyperparameters_object._dataset.filename_samples).split('_test')[0]

        if model == 'svm':
            kernel = self.__classifier_hyperparameters_object._classifier.kernel
            filename = f'{prefix}_{model}_{kernel}.png'
        
        elif model == 'lda' or model == 'qda':
            solver = self.__classifier_hyperparameters_object._classifier.solver
            filename = f'{prefix}_{model}_{solver}.png'
        
        else:
            raise ValueError(f'In defining output plot filename. Got {model}, when expected svm, lda or qda')

        return (
            GlobalPaths.OUTPUT_FILES / f'plot_{model}' /
            filename
        )

    def main(self):
        if self.__classifier_hyperparameters_object._classifier.mode == 'train':
            self.__train()
            self.__save_model(self.__define_model_filename())
        
        elif self.__classifier_hyperparameters_object._classifier.mode == 'test':
            self.__model = self.__load_model(self.__classifier_hyperparameters_object._classifier.saved_model_name)
            self.__evaluate()

if __name__ == '__main__':
    classifier = Classifier()
    classifier.main()