import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import numpy as np
import yaml
import sys
import os
import matplotlib.pyplot as plt
from pandas import read_csv
from resnet.resnet_class import ResidualBlock, ResNet, InputVariablesResnet
from vgg.vgg_class import VGG19, InputVariablesVGG19
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import List
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import get_today_string, GlobalPaths

sys.path.insert(1, str(GlobalPaths.DATASET))
from dataset import Dataset

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

    def plot_metrics(self, output_path: str, model_name: str, optimizer: str, num_epochs: int, df_name : str):
        """
          Salva i plot delle metriche con un nome file coerente con lo stile:
          YYYY-MM-DD_<model_name>_<optimizer>_<num_epochs>_<df_name>_<metric>.png

          In questo modo, rendo il formato del filename di output coerente con quello
          relativo alle caratteristiche estratte, salvate in features_step1_cnn.
        """
        os.makedirs(output_path, exist_ok=True)  # crea la directory se non esiste
        today = get_today_string()
        metrics = ['loss', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(self.epochs, getattr(self, metric), label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'Training {metric.capitalize()} Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = f"{today}_{model_name}_{optimizer}_{num_epochs}_{df_name}_{metric}.png"
            plt.savefig(os.path.join(output_path, filename), dpi=300)
            plt.close()

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
    _metrics_output_path: str

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
            _metrics_output_path=config.get('metrics_output_path', GlobalPaths.OUTPUT_FILES / 'training_metrics')
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
        self.__training_data_loader = self.__dataset.get_training_data_loader(batch_size = self.__training_hyperparameters._batch_size)
        # Init loss function
        self.__criterion = self.__init_loss()
        # Init optimizer
        self.__optimizer = self.__init_optimizer()
        # Output: feature vectors
        self.__extracted_features = []
        self.__extracted_labels = []

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
          num_pos = self.__dataset.get_training_set_labels().sum()                                # Number of positive samples (class 1: planet)
          num_neg = len(self.__dataset.get_training_set_labels()) - num_pos                       #           negative         (class 0: not-planet)
          pos_weight = (num_neg / num_pos).clone().detach().to(device)  # Weight applied to the loss function (ICF)
          return pos_weight
        else:
          class_counts = torch.bincount(self.__dataset.get_training_set_labels(), minlength=self.__training_hyperparameters._num_classes) # Compute class frequency
          class_weights = len(self.__dataset.get_training_set_labels()) / (self.__training_hyperparameters._num_classes * class_counts)   # ICF method
          class_weights = class_weights.float().to(device) # Convert into tensor format
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

    def __feed_forward_pass(self, batch_x):
        # Feed-forward pass
        if 'resnet' in self.__training_hyperparameters._model_name:
          # The order is swapped wrt VGG19 as Resnet class contains both feature extraction and classification in it
          outputs = self.__model(batch_x)
          features = self.__model.get_feature_extraction_output() 

        elif 'vgg' in self.__training_hyperparameters._model_name:
          features = self.__model.get_feature_extraction_output(batch_x)
          outputs = self.__model.get_classification_output(features)

        else:
          raise ValueError(f'Got {self.__training_hyperparameters._model_name}, but work with vgg and resnet only.')
        
        return features, outputs

    def __save_extracted_feature_vectors(self):
        """
          Salva i vettori di caratteristiche, con i relativi labels, estratti durante l'ultima epoca di training.
          I dati vengono salvati in due file .npy separati, uno per le features e uno per i labels.
          
          Il percorso del file è costruito dinamicamente in base alla data corrente (formato YYYY-MM-DD),
          al nome del modello, all'ottimizzatore utilizzato e al numero di epoche di training.

          I file vengono salvati nella directory definita da GlobalPaths.FEATURES_STEP1_CNN.
        """
        # filepath structure: features_step1_cnn/<YYYY-MM-DD>_<model_name>_<optimizer>_<num_epochs>_<catalog_name>_<features/labels>.npy
        today = get_today_string()
        filepath_base = (
          GlobalPaths.FEATURES_STEP1_CNN / 
          f'{today}_{self.__training_hyperparameters._model_name}_{self.__training_hyperparameters._optimizer}_{self.__training_hyperparameters._num_epochs}_{self.__dataset.get_catalog_name()}_'
        )
        all_features = np.concatenate(self.__extracted_features, axis=0)
        all_labels = np.concatenate(self.__extracted_labels, axis=0)

        np.save(filepath_base.with_name(filepath_base.name + 'features.npy'), all_features)
        np.save(filepath_base.with_name(filepath_base.name + 'labels.npy'), all_labels)
        print(f'Features and labels saved to {filepath_base}features.npy and {filepath_base}labels.npy')      

    def train(self):
        for epoch in tqdm(range(self.__training_hyperparameters._num_epochs), desc="Training Epochs", unit="epoch"):
          self.__model.train()    # setta il modello in modalità training
          running_loss = 0.0

          all_labels = []
          all_outputs = []
          all_probs = []

          # Iterate on batch during i-th training step 
          for batch_x, batch_y in self.__training_data_loader:
            # As batch_x shape is torch.Size([batch_size, 201]), we need to convert to (batch_size, in_channels=1, signal_length=201)
            # in order Resnet is able to feed-forwardly process it. For this reason, we apply the unsqueeze() method to every batch
            # before computation.  
            batch_x = batch_x.unsqueeze(1) 
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # put it on the same device where the model is
            
            self.__optimizer.zero_grad()                              # init gradients
            if self.__training_hyperparameters._num_classes > 1:
              batch_y = batch_y.long()                                # torch.nn.CrossEntropyLoss() requires y_true to be torch.long
            
            # Feed-forward pass
            features, outputs = self.__feed_forward_pass(batch_x)
            
            # Update loss
            if self.__training_hyperparameters._num_classes > 1:
              loss = self.__criterion(outputs, batch_y.squeeze())     # remove any additional dimension from labels
              probs = F.softmax(outputs, dim=1)                       # probabilities provided for each sample. required by auc_roc in multi-class scenarios
              predictions = torch.argmax(probs, dim=1)                # discrete predictions provided for each sample: {0, 1, 2}
            else:
              loss = self.__criterion(outputs, batch_y.unsqueeze(1).float())
              predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            
            # Backpropagation
            loss.backward()
            self.__optimizer.step()
            # Update the loss function and the other metrics. Use methods from the class TrainingMetrics
            running_loss += loss.item() * batch_x.size(0)

            all_labels.extend(batch_y.detach().cpu().numpy())
            all_outputs.extend(predictions.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
          
          # Training on i-th batch has ended. Compute epoch metrics
          epoch_loss = running_loss / self.__dataset.get_training_set_length()
          precision = precision_score(all_labels, all_outputs, average='macro', zero_division=0)
          recall = recall_score(all_labels, all_outputs, average='macro', zero_division=0)
          f1 = f1_score(all_labels, all_outputs, average='macro', zero_division=0)
          if self.__training_hyperparameters._num_classes > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
          else:
            auc = roc_auc_score(all_labels, all_outputs)

          # Log metrics
          self.__training_metrics.log(epoch, epoch_loss, precision, recall, f1, auc)
          self.__training_metrics.print_last()

          # Save feature vectors during last epoch
          if epoch == self.__training_hyperparameters._num_epochs - 1:
            self.__extracted_features.append(features.detach().cpu().numpy())
            self.__extracted_labels.append(batch_y.detach().cpu().numpy())

        print("\nTraining completed.")        
        # Plot training metrics once training is completed. Use methods from the class TrainingMetrics
        self.__training_metrics.plot_metrics(
          output_path=GlobalPaths.OUTPUT_FILES / self.__training_hyperparameters._metrics_output_path,
          model_name=self.__training_hyperparameters._model_name,
          optimizer=self.__training_hyperparameters._optimizer,
          num_epochs=self.__training_hyperparameters._num_epochs,
          df_name=self.__dataset.get_catalog_name()
          )
        
        # Concatenate and save feature vectors and labels
        self.__save_extracted_feature_vectors()

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
      with open(GlobalPaths.CONFIG / 'config_dataset.yaml', 'r') as fd:
          config_fd = yaml.safe_load(fd)
      # Carica il dataset CSV
      df = read_csv(GlobalPaths.CSV / config_fd['dataset_filename'])
      # Istanzia la classe Dataset ed esegue tutte le operazioni
      return Dataset(df, config_fd)

    def __init_model(self):
      # 2. Initialize model architecture with config_vgg(resnet).yaml.
      with open(GlobalPaths.CONFIG / 'config_feature_extractor.yaml', 'r') as fe:
          config_fe = yaml.safe_load(fe)
      
      if config_fe['model_name'] == 'vgg':
          # load data from config_vgg.yaml
          self.__model_hyperparameters_object = InputVariablesVGG19.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_vgg.yaml')
          
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
          self.__model_hyperparameters_object = InputVariablesResnet.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_resnet.yaml')

          # Create the model
          self.__model = ResNet(
            ResidualBlock, 
            self.__model_hyperparameters_object.get_resnet_layers_num(),
            self.__model_hyperparameters_object.get_fc_output_size()
            ).to(device)
          print(self.__model)
    
    def __init_training_hyperparameters(self):
        self.__training_hyperparameters_object = InputVariablesModelTraining.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_feature_extractor.yaml')

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