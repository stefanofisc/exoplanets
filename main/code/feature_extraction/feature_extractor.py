import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import numpy as np
import gc
import yaml
import sys
from pandas import read_csv
from resnet.resnet_class import PathConfigResnet, ResidualBlock, ResNet, InputVariablesResnet
from vgg.vgg_class import PathConfigVGG19, VGG19, InputVariablesVGG19
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from typing import List

sys.path.insert(0, '/home/stefanofiscale/Desktop/exoplanets/main/code/dataset/')
from dataset import PathConfigDataset, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing training on {device}")

class ModelInspector:
  def __init__(self, model):
    self.__model = model

  def count_trainable_params(self):
    return sum(p.numel() for p in self.__model.parameters() if p.requires_grad)

  def print_layer_params(self):
    print(f"{'Layer':<30} {'Param Count':<20}")
    print("="*50)
    for name, param in self.__model.named_parameters():
      print(f"{name:<30} {param.numel():<20}")

@dataclass
class TrainingMetrics:
    epochs: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1: List[float] = field(default_factory=list)
    auc_roc: List[float] = field(default_factory=list)

    def log(self, epoch, loss, precision, recall, f1, auc):
        """Aggiunge i valori di una singola epoca"""
        self.epochs.append(epoch)
        self.loss.append(loss)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.auc_roc.append(auc)

    def print_last(self):
        """Stampa i valori dell’ultima epoca"""
        print(f"Epoch {self.epochs[-1]} — Loss: {self.loss[-1]:.4f}, Precision: {self.precision[-1]:.3f}, Recall: {self.recall[-1]:.3f}, F1: {self.f1[-1]:.3f}, AUC: {self.auc_roc[-1]:.3f}")

    def plot_metrics(self):
        metrics = ['loss', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in metrics:
            plt.plot(self.epochs, getattr(self, metric), label=metric)

        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Training metrics')
        plt.grid(True)
        plt.tight_layout()
        #plt.savefig(#TODO. Path to be defined)

# 2025-05-31. Questa classe va rivista per bene in funzione del refactoring.
class MetricTracker:
  def __init__(self, device, num_classes):
    self.__losses = []
    self.__accuracies = []
    self.__average = 'macro'
    self.__task = ''
    self.__num_classes = num_classes
    if num_classes == 1 or num_classes == 2:
      self.__task = 'binary'
    else:
      self.__task = 'multiclass'
    
    self.__precision_metric = torchmetrics.Precision(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__recall_metric = torchmetrics.Recall(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__f1_metric = torchmetrics.F1Score(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__auc_roc_metric = torchmetrics.AUROC(task=self.__task, num_classes=self.__num_classes).to(device)
    self.__precision_recall_curve_metric = torchmetrics.PrecisionRecallCurve(task=self.__task, num_classes=self.__num_classes).to(device)  # return a tensor containing three lists: recall, precision, thresholds


  def _compute(self):
    """
      Compute evaluation metrics during model training
    """
    precision               = self.__precision_metric.compute()
    recall                  = self.__recall_metric.compute()
    f1                      = self.__f1_metric.compute()
    auc_roc                 = self.__auc_roc_metric.compute()
    precision_recall_curve  = self.__precision_recall_curve_metric.compute()
    # Reset metrics for next computation
    self.__precision_metric.reset()
    self.__recall_metric.reset()
    self.__f1_metric.reset()
    self.__auc_roc_metric.reset()
    self.__precision_recall_curve_metric.reset()

    return precision.item(), recall.item(), f1.item(), auc_roc.item(), precision_recall_curve

  def _update(self, outputs, labels, loss):
    """
      This method compute and update the evaluation metrics at the end of each training epoch.
      Input:
        - outputs: the predictions of the model. type: torch.Tensor. shape: [batch_size, num_classes];
        - labels: the true labels of the examples. type: torch.Tensor. shape: [batch_size];
        - loss: loss function value. type: torch.Tensor.
      Further information:
        About _precision_recall_curve_metric:
        Since outputs.shape:[batch_size, num_classes] and labels.shape:[batch_size] you have to remove the extra dimension 
        from outputs (or add a dimension to labels). Moreover, torchmetrics.PrecisionRecallCurve expects the 
        labels tensor contains integer values (typically torch.int or torch.long). To be sure labels is of the
        correct data type (in some cases it could be torch.float32 with i-th label value = 1.0), we cast the tensor.
    """
    self.__losses.append(loss.item())

    if self.__num_classes == 1:
      # Binary classification
      preds = torch.round(torch.sigmoid(outputs).squeeze(-1)) # Remove the extra dimension from outputs tensor so that it has the same shape of labels. Store values in preds.
      accuracy = (preds == labels.unsqueeze(1)).float().mean()
    else:
      # Multi-class classification
      # Computing accuracy for multi-class classification requires the following steps:
      # 1. Use torch.argmax on model's predictions to achieve the index of the class with highest likelihood for each sample;
      # 2. Compare y_pred with y_true
      preds = torch.argmax(outputs, dim=1)
      accuracy = (preds == labels).float().mean()

    self.__accuracies.append(accuracy.item())
    self.__precision_metric(preds, labels)
    self.__recall_metric(preds, labels)
    self.__f1_metric(preds, labels)
    self.__auc_roc_metric(outputs, labels)    # AUC-ROC and PR curve require the logits or probabilities, then use the tensor outputs
    self.__precision_recall_curve_metric(outputs.squeeze(dim=1), labels.long())  

  def _plot(self, x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='.')
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=1200)
    plt.close()

  def _plot_training_metrics(self, epoch, epoch_loss, precision, recall, f1, auc_roc, path):
    """
      Plot the values for each evaluation metrics during training
      Input:
        - epoch_loss, precision, recall, f1, auc_roc. type: list
    """
    self._plot(epoch, epoch_loss, 'Epoch', 'Epoch loss', 'Loss function', path + 'training_epoch.png')
    self._plot(epoch, precision, 'Epoch', 'Precision', 'Training precision', path+'training_precision.png')
    self._plot(epoch, recall, 'Epoch', 'Recall', 'Training recall', path+'training_recall.png')
    self._plot(epoch, f1, 'Epoch', 'F1-score', 'Training f1-score', path+'training_f1.png')
    self._plot(epoch, auc_roc, 'Epoch', 'AUROC', 'Training Area under ROC', path+'training_aucroc.png')

  def _compute_metrics_from_confusion_matrix(self, confusion_matrix):
    """
    Calcola precision, recall, f1-score e misclassification rate per ogni classe
    data una matrice di confusione.
    
    Parameters:
    -----------
    confusion_matrix : numpy.ndarray
        Matrice di confusione di dimensione NxN, dove N è il numero di classi.
        Gli elementi confusion_matrix[i, j] rappresentano il numero di elementi
        della classe i predetti come classe j.
    
    Returns:
    --------
    dict
        Dizionario contenente le metriche per ogni classe:
        {
            'precision': array con precision per ogni classe,
            'recall': array con recall per ogni classe,
            'f1_score': array con f1-score per ogni classe,
            'misclassification_rate': array con misclassification rate per ogni classe
        }
    """
    # Verifica che la matrice sia quadrata
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Error: confusion matrix must be squared!")
    
    # Verifica che la matrice sia in formato np.array
    if not isinstance(confusion_matrix, np.ndarray):
        print("Warning: input matrix type is not numpy ndarray. Casting into np.array().")
        confusion_matrix = np.array(confusion_matrix)
    
    num_classes = confusion_matrix.shape[0]
    total_samples = np.sum(confusion_matrix)
    
    # Inizializzazione degli array per le metriche
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    misclassification_rate = np.zeros(num_classes)
    
    # Calcolo delle metriche per ogni classe
    for i in range(num_classes):
        # True Positives: elementi della classe i correttamente classificati
        tp = confusion_matrix[i, i]
        
        # False Positives: elementi di altre classi classificati come classe i
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False Negatives: elementi della classe i classificati come altre classi
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # True Negatives: elementi di altre classi correttamente classificati come non-i
        tn = total_samples - tp - fp - fn
        
        # Calcolo precision
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calcolo recall
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calcolo F1-score
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        # Calcolo misclassification rate
        misclassification_rate[i] = (fp + fn) / total_samples
    
    # Creazione del dizionario di risultati
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'misclassification_rate': misclassification_rate
    }
    
    return results

  def _print_metrics_from_confusion_matrix(self, confusion_matrix, class_names=None, digits=3):
      """
      Stampa una tabella formattata con le metriche calcolate dalla matrice di confusione.
      
      Parameters:
      -----------
      confusion_matrix : numpy.ndarray
          Matrice di confusione di dimensione NxN.
      class_names : list, optional
          Lista dei nomi delle classi. Se non fornito, usa 'Classe 1', 'Classe 2', ecc.
      digits : int, optional
          Numero di cifre decimali da visualizzare. Default: 3.
      """
      if class_names is None:
          class_names = [f'Classe {i+1}' for i in range(confusion_matrix.shape[0])]
      
      # Calcolo delle metriche
      metrics = self._compute_metrics_from_confusion_matrix(confusion_matrix)
      
      # Stampa della tabella
      print(f"{'Classe':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Misclass Rate':<15}")
      print("-" * 60)
      
      for i, class_name in enumerate(class_names):
          print(f"{class_name:<10} {metrics['precision'][i]:<12.{digits}f} "
                f"{metrics['recall'][i]:<12.{digits}f} "
                f"{metrics['f1_score'][i]:<12.{digits}f} "
                f"{metrics['misclassification_rate'][i]:<15.{digits}f}")

  def compute_evaluation_metrics(self, all_preds, all_probs, all_labels):
    # Metrics computation with scikit-learn
    #NOTE. Experimental: weighted average during evaluation
    self.__average = 'weighted' #weighted allows you to calculate metrics for each label and find their average weighted by support (the number of true instancs for each label). Account for class imbalance. This can result in an F-score that is not between precision and recall.

    # Precision, recall and F1 score
    try:
      precision = precision_score(all_labels, all_preds, average=self.__average)  # NOTE. prima era 'binary' per binary classification. Check for errors
      recall    = recall_score(all_labels, all_preds, average=self.__average)
      f1        = f1_score(all_labels, all_preds, average=self.__average)
    except Exception as e0:
      print(f'In m1_class.py --> Class MetricTracker --> compute_evaluation_metrics() --> Error while computing precision, recall and f-score.\nError:{e0}')

    # Area under ROC
    if self.__num_classes == 1:
      auc_roc = roc_auc_score(all_labels, all_preds)
    else:
      # In multi-class classification this metric requires the probabilities score the model returned for each class. For this reason, we pass all_probs.
      # 'ovr' stands for One-vs-rest. Computes the AUC of each class against the rest. This treats the multiclass case in the same way as the multilabel case. 
      # Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
      try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=self.__average)
        print(f'AUROC: {auc_roc:.4f}')  # su riga separata rispetto a precision, recall ed fscore, altrimenti puoi avere errore UnboundLocalError: local variable 'auc_roc' referenced before assignment nel caso in cui il blocco try{} di auc_roc va in errore.
      except Exception as e1:
        print(f'In m1_class.py --> Class MetricTracker --> compute_evaluation_metrics() --> Error while plotting ROC curve in task:multiclass.\nError:{e1}')

    # Print weighted metrics
    print('Weighted metrics:')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Confusion matrix and metrics without weighting
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion matrix:')
    print(f'{np.array(conf_matrix)}')
    print("\nMetrics per class:")
    if self.__num_classes > 1:
      self._print_metrics_from_confusion_matrix(conf_matrix, ['AFP','PC','NTP'])
    else:
      self._print_metrics_from_confusion_matrix(conf_matrix, ['NPC','PC'])
    
    # Accesso diretto alle metriche
    metrics = self._compute_metrics_from_confusion_matrix(conf_matrix)
    print("\nDirect access to metrics:")
    for metric_name, values in metrics.items():
        print(f"{metric_name}: {values}")

  def compute_confmat_files_with_toi(self, all_pred, all_labels, x_test_koi, path_to_save):
    """ 
    ------------------------------------------------------------------------------
      Input: tre vettori numpy
      - all_labels: etichette reali (valori binari: 0 o 1)
      - all_pred: predizioni del modello (valori binari: 0 o 1)
      - x_test_koi: identificatori associati a ciascun elemento del test set (es. KOI)
      
      Output: salva in quattro file di testo i KOI corrispondenti a:
      - true_positives.txt: etichetta = 1, predizione = 1
      - true_negatives.txt: etichetta = 0, predizione = 0
      - false_positives.txt: etichetta = 0, predizione = 1
      - false_negatives.txt: etichetta = 1, predizione = 0
      ------------------------------------------------------------------------------
    """
    # (0) Controllo sulla lunghezza dei vettori 
    if not (len(all_labels) == len(all_pred) == len(x_test_koi)):
      raise ValueError(
          f"I vettori devono avere la stessa lunghezza, ma sono:\n"
          f"- all_labels: {len(all_labels)}\n"
          f"- all_pred: {len(all_pred)}\n"
          f"- x_test_koi: {len(x_test_koi)}"
      )
    # Converti i vettori
    print("--- Converto i vettori in numpy.ndarray ---")
    all_labels = np.array(all_labels).flatten()
    all_pred = np.array(all_pred).flatten()
    x_test_koi = np.array(x_test_koi).flatten()
    print("--- Verifico tipo e shape dei vettori ---")
    print(type(all_labels), all_labels.shape)
    print(type(all_pred), all_pred.shape)
    print(type(x_test_koi), x_test_koi.shape)
    try:
      # (1) Estrai i valori dei TOI(KOI) per ogni casella della matrice di confusione
      true_positives = x_test_koi[(all_labels == 1) & (all_pred == 1)]
      true_negatives = x_test_koi[(all_labels == 0) & (all_pred == 0)]
      false_positives = x_test_koi[(all_labels == 0) & (all_pred == 1)]
      false_negatives = x_test_koi[(all_labels == 1) & (all_pred == 0)]
    except Exception as e1:
      print(f'In compute_confmat_files_with_toi() --> Errore durante il filtraggio: {e1}')
    
    try:
      # (2) Salva i risultati
      np.savetxt(path_to_save+"true_positives.txt", true_positives, fmt='%d')
      np.savetxt(path_to_save+"true_negatives.txt", true_negatives, fmt='%d')
      np.savetxt(path_to_save+"false_positives.txt", false_positives, fmt='%d')
      np.savetxt(path_to_save+"false_negatives.txt", false_negatives, fmt='%d')
    except Exception as e2:
      print(f'In compute_confmat_files_with_toi() --> Errore durante il salvataggio: {e2}')


  def plot_prcurve(self, all_labels, all_preds, path):
    # Compute and plot the precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    if path:
      plt.savefig(path+'prcurve.png', dpi=1200)
    else:
      plt.show()
    plt.close()
  
  def plot_roc(self, all_labels, all_preds, path):
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    self._plot(fpr, tpr, 'False positive rate', 'True positive rate', 'ROC curve', path+'roc_curve.png')

  def print_metrics(self):
    """
      Print evaluation metrics. 
      #NOTE. Actually, you don't use this method anywhere
    """
    precision, recall, f1, auc_roc, precision_recall_curve = self.__compute()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC-ROC: {auc_roc}")
    
  def get_precision(self):
    return self.__precision_metric.compute()

  def get_recall(self):
    return self.__recall_metric.compute()

  def get_f1(self):
    return self.__f1_metric.compute()

  def get_auc_roc(self):
    return self.__auc_roc_metric.compute()

  def get_precision_recall_curve(self):
    return self.__precision_recall_curve_metric.compute()

  def get_losses(self):
    return self.__losses

  def get_accuracies(self):
    return self.__accuracies


@dataclass
class InputVariablesModelTraining:
    # define type of input data
    _model_name: str
    _optimizer: str
    _learning_rate: float
    _num_epochs: int
    _batch_size: int
    _num_classes: int
    _weight_decay: float
    _momentum: float
    pass
    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _model_name=config['model_name'],
            _optimizer=config['optimizer'],
            _learning_rate=config['learning_rate'],
            _num_epochs=config['num_epochs'],
            _batch_size=config['batch_size'],
            _num_classes=config['num_classes'],
            _weight_decay=config.get('weight_decay', None),   # necessario solo se optimizer = SGD
            _momentum=config.get('momentum', None),           # necessario solo se optimizer = SGD
            )
    
    # define get and set methods
    pass

class Model:
    def __init__(self, dataset, model, model_hyperparameters_object, training_hyperparameters_object):
        """
            Costruttore della classe Model. 
            Input:
                - dataset: dataset di input. Oggetto della classe Dataset
                - model: modello di input. Può essere un oggetto della classe Resnet o VGG;
                - model_hyperparameters_object: iperparametri per definire l'architettura del modello;
                - training_hyperparameters_object: iperparametri di training;
        """
        # Init model architecture
        self.__model = model
        self.__model.to(device)
        self.__model_hyperparameters = model_hyperparameters_object
        self.__training_hyperparameters = training_hyperparameters_object
        self.__training_metrics = TrainingMetrics()
        # Init dataset and training-test tensors
        self.__dataset = dataset
        #NOTE. Commento dati train e test perché per ora non mi servono singolarmente.
        #self.__X_train, self.__y_train, self.__X_test, self.__y_test = self.__dataset.get_training_test_samples()
        self.__training_data_loader = self.__dataset.get_training_data_loader(batch_size = self.__training_hyperparameters._batch_size)
        # Init MetricTracker object to keep track of training-test metrics
        self.__metric_tracker = MetricTracker(device, self.__model_hyperparameters._fc_output_size)
        # Init loss function
        self.__criterion = self.__init_loss()
        # Init optimizer
        self.__optimizer = self.__init_optimizer()

    
    def __init_loss(self):
        """
          Method for loss function initialization. Select one between Binary Cross-Entropy and Cross-Entropy depending on
          the classification task at hand.
        """
        if self.__model_hyperparameters._fc_output_size == 1:
          return nn.BCEWithLogitsLoss(pos_weight = self.__init_class_weighting())
        else:
          return nn.CrossEntropyLoss(weight = self.__init_class_weighting())

    def __init_class_weighting(self):
        """
          Method for class weighting initialization. Use the Inverse Class Frequency (ICF) method.
          Output:
            - class_weights: for multi-class classification, a torch.Tensor containing the weights computed for each class
        """
        if self.__model_hyperparameters._fc_output_size == 1:
          num_pos = self.__y_train.sum()                                # Number of positive samples (class 1: planet)
          num_neg = len(self.__y_train) - num_pos                       #           negative         (class 0: not-planet)
          pos_weight = (num_neg / num_pos).clone().detach().to(device)  # Weight applied to the loss function (ICF)
          return pos_weight
        else:
          class_counts = torch.bincount(self.__y_train, minlength=self.__training_hyperparameters._num_classes) # Compute class frequency
          class_weights = len(self.__y_train) / (self.__training_hyperparameters._num_classes * class_counts)   # ICF method
          class_weights = class_weights.float() # Convert into tensor format
          return class_weights

    def __init_optimizer(self):
        if self.__training_hyperparameters._optimizer == 'adam':
          print('\nOptimizer: Adam')
          return optim.Adam(self.__model.parameters(), lr=self.__training_hyperparameters._learning_rate)
        elif self.__training_hyperparameters._optimizer == 'sgd':
          print('\nOptimizer: SGD')
          return optim.SGD(
            self.__model.parameters(),
            lr = self.__training_hyperparameters._learning_rate,
            weight_decay = self.__training_hyperparameters._weight_decay,
            momentum = self.__training_hyperparameters._momentum
          )
        else:
          raise ValueError(f'Got {self.__training_hyperparameters._optimizer}, but work with Adam and Stochastic Gradient Descent optimizers only.\n Please set adam or sgd to train the model.')

    def train(self):
        for epoch in range(self.__training_hyperparameters._num_epochs):
          self.__model.train()    # setta il modello in modalità training
 
          for batch_x, batch_y in self.__training_data_loader
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # iterate on batch. before computation, put it on the same device where the model is
            
            self.__optimizer.zero_grad()                              # init gradients
            if self.__training_hyperparameters._num_classes > 1:
              batch_y = batch_y.long()                                # torch.nn.CrossEntropyLoss() requires y_true to be torch.long
            
            # Feed-forward pass
            features = self.__model.get_feature_extraction_output(batch_x)   #TODO. Check why u don't use batch_size as input for resnet
            if self.__training_hyperparameters._model_name == 'resnet':
              #TODO. Fare in modo che resnet e vgg abbiano stesso metodo feed-forward per ottenere output
              outputs = self.__model()
            elif self.__training_hyperparameters._model_name == 'vgg':
              outputs = self.__model.get_classification_output(features)
            else:
              raise ValueError(f'Got {self.__training_hyperparameters._model_name}. Work with vgg and resnet only.')
            
            # Update loss
            if self.__training_hyperparameters._num_classes > 1:
              loss = self.__criterion(outputs, batch_y.squeeze())     # remove any additional dimension from labels
            else:
              loss = self.__criterion(outputs, batch_y.unsqueeze(1).float())
            
            # Backpropagation
            loss.backward()
            self.__optimizer.step()
            # Update the loss function and the other metrics
            # TODO. Codice training non ancora terminato




          
        
    def evaluate(self):
        pass
        
    def __del__(self):
        print('\nDestructor called for the class Model')


class FeatureExtractor:
    def __init__(self):
      """
        Costruttore della classe FeatureExtractor. Carica l'oggetto dataset e l'oggetto model
      """
      # Private attributes of the class
      self.__dataset_handler = self.__init_dataset()
      self.__model_hyperparameters_object = None
      self.__model = None
      self.__training_hyperparameters_object = None
      self.__init_model()
      self.__init_training_hyperparameters()
    
    def __init_dataset(self):
      # 1. Initialize Dataset object with config_dataset.yaml
      with open(PathConfigDataset.BASE_DATASET_MODULE / 'config_dataset.yaml', 'r') as fd:
          config_fd = yaml.safe_load(fd)
      # Carica il dataset CSV
      df = read_csv(PathConfigDataset.CSV / config_fd['dataset_filename'])
      # Istanzia la classe Dataset ed esegue tutte le operazioni
      return Dataset(df, config_fd)

    def __init_model(self):
      # 2. Initialize model architecture with config_vgg(resnet).yaml.
      with open('config_feature_extractor.yaml', 'r') as fe:
          config_fe = yaml.safe_load(fe)
      
      if config_fe['model_name'] == 'vgg':
          # load data from config_vgg.yaml
          self.__model_hyperparameters_object = InputVariablesVGG19.get_input_hyperparameters(PathConfigVGG19.VGG / 'config_vgg.yaml')
          
          # Create the model architecture
          self.__model = VGG19(
              self.__model_hyperparameters_object.get_psz(),
              self.__model_hyperparameters_object.get_pst(),
              self.__model_hyperparameters_object.get_fc_layers_num(),
              self.__model_hyperparameters_object.get_fc_units(),
              self.__model_hyperparameters_object.get_fc_output_size()
              )
          print(self.__model)
      else:
          # load data from config_resnet.yaml
          self.__model_hyperparameters_object = InputVariablesResnet.get_input_hyperparameters(PathConfigResnet.RESNET / 'config_resnet.yaml')

          # Create the model
          self.__model = ResNet(
            ResidualBlock, 
            self.__model_hyperparameters_object.get_resnet_layers_num(),
            self.__model_hyperparameters_object.get_fc_output_size()
            ).to(device)
          print(self.__model)
    
    def __init_training_hyperparameters(self):
        self.__training_hyperparameters_object = InputVariablesModelTraining.get_input_hyperparameters('config_feature_extractor.yaml')

    def __feature_extraction(self):
      # Initialize Model object
      model = Model(
        self.__dataset_handler, 
        self.__model, 
        self.__model_hyperparameters_object, 
        self.__training_hyperparameters_object
        )
      # Train the model
      model.train()

    def main(self):
      self.__feature_extraction()

    def __del__(self):
      print('\nDestructor called for the class FeatureExtractor')

if __name__ == '__main__':
  feature_extractor = FeatureExtractor()
  feature_extractor.main()