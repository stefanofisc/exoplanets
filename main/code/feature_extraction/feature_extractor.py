import torch
import torch.nn               as nn
import torch.nn.functional    as F
import torch.optim            as optim
import numpy                  as np
import yaml
import sys
import os
from  pandas                  import read_csv
from  resnet.resnet_class     import ResidualBlock, ResNet, InputVariablesResnet
from  vgg.vgg_class           import VGG19, InputVariablesVGG19
from  dataclasses             import dataclass#, field
from  sklearn.metrics         import precision_score, recall_score, f1_score, roc_auc_score
from  tqdm                    import tqdm
from  pathlib                 import Path
from  typing                  import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from  utils                   import get_today_string, GlobalPaths, get_device, TrainingMetrics

sys.path.insert(1, str(GlobalPaths.DATASET))
from  dataset                 import Dataset

device = get_device()
print(f"Executing training on {device}")

"""
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
"""

@dataclass
class InputVariablesModelTraining:
    # define type of input data
    _mode:              str
    _model_name:        str
    _optimizer:         str
    _learning_rate:     float
    _num_epochs:        int
    _batch_size:        int
    _num_classes:       int
    _weight_decay:      float
    _momentum:          float
    _save_model:        bool
    _purpose:           str
    _saved_model_name:  Optional[str] = ''

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _mode             = config['mode'],                             # the only mandatory input variable: values={train,test}
            _model_name       = config.get('model_name', None),
            _optimizer        = config.get('optimizer', None),
            _learning_rate    = config.get('learning_rate', None),
            _num_epochs       = config.get('num_epochs', None),
            _batch_size       = config.get('batch_size', 64),
            _num_classes      = config.get('num_classes', None),
            _weight_decay     = config.get('weight_decay', None),   # necessario solo se optimizer = SGD
            _momentum         = config.get('momentum', None),           # necessario solo se optimizer = SGD
            _save_model       = config.get('save_model', False),
            _purpose          = config.get('purpose', 'TBD'),
            _saved_model_name = config.get('saved_model_name', '')
            )
    
    # define get and set methods
    def get_mode(self):
      return self._mode
    
    pass

class Model:
    def __init__(self, dataset, model, model_hyperparameters_object, training_test_hyperparameters_object):
        """
            Costruttore della classe Model. 
            Input:
                - dataset: dataset di input. Oggetto della classe Dataset
                - model: modello di input. Può essere un oggetto della classe Resnet o VGG;
                - model_hyperparameters_object: iperparametri per definire l'architettura del modello;
                - training_test_hyperparameters_object: iperparametri di training / test;
        """
        # Init model architecture
        self.__model                      = model
        self.__model.to(device)
        self.__model_hyperparameters      = model_hyperparameters_object
        # Init dataset object
        self.__dataset                    = dataset
        
        self.__training_hyperparameters   = None
        self.__test_hyperparameters       = None

        if training_test_hyperparameters_object.get_mode() == 'train':
          # Load training set (PyTorch DataLoader object)
          #NOTE EXPERIMENT
          #self.__training_data_loader      = self.__dataset.get_full_data_loader(batch_size = training_test_hyperparameters_object._batch_size)
          #NOTE END EXPERIMENT
          #NOTE DECOMMENTA riga sotto
          self.__training_data_loader     = self.__dataset.get_training_data_loader(batch_size = training_test_hyperparameters_object._batch_size)
          
          self.__training_hyperparameters = training_test_hyperparameters_object
          self.__training_metrics         = TrainingMetrics()

          self.__optimizer                = self.__init_optimizer()
        else:
          # Load test set (a PyTorch DataLoader object)
          self.__test_data_loader         = self.__dataset.get_test_data_loader(batch_size = training_test_hyperparameters_object._batch_size)
          self.__test_hyperparameters     = training_test_hyperparameters_object
        
        # Init loss function
        self.__criterion = self.__init_loss()

        # Output: feature vectors
        self.__extracted_features = []
        self.__extracted_labels   = []

    def __init_loss(self):
        """
          Method for loss function initialization. Select one between Binary Cross-Entropy and Cross-Entropy depending on
          the classification task at hand.
        """
        if self.__training_hyperparameters:
          # Training mode
          if self.__model_hyperparameters._fc_output_size == 1:
            return nn.BCEWithLogitsLoss(pos_weight = self.__init_class_weighting())
          else:
            return nn.CrossEntropyLoss(weight = self.__init_class_weighting())
        else:
          # Test mode. Init loss without class weighting
          # NOTE. 2025-06-11
          #       We initialize loss function for multi-class classification.
          #       Future implementations will consider the initialization of binary loss in test mode
          return nn.CrossEntropyLoss()

    def __init_class_weighting(self):
        """
          Method for class weighting initialization. Use the Inverse Class Frequency (ICF) method.
          Output:
            - class_weights: for multi-class classification, a torch.Tensor containing the weights computed for each class
        """
        split = 'train'
        if self.__model_hyperparameters._fc_output_size == 1:
          num_pos       = self.__dataset.get_training_test_set_labels(split).sum()                                # Number of positive samples (class 1: planet)
          num_neg       = len(self.__dataset.get_training_test_set_labels(split)) - num_pos                       #           negative         (class 0: not-planet)
          pos_weight    = (num_neg / num_pos).clone().detach().to(device)  # Weight applied to the loss function (ICF)
          return        pos_weight
        else:
          class_counts  = torch.bincount(self.__dataset.get_training_test_set_labels(split), minlength=self.__training_hyperparameters._num_classes) # Compute class frequency
          class_weights = len(self.__dataset.get_training_test_set_labels(split)) / (self.__training_hyperparameters._num_classes * class_counts)   # ICF method
          class_weights = class_weights.float().to(device) # Convert into tensor format
          return        class_weights

    def __init_optimizer(self):
        if self.__training_hyperparameters._optimizer == 'adam':
          #print('\nOptimizer: Adam')
          return optim.Adam(self.__model.parameters(), lr=self.__training_hyperparameters._learning_rate)
        elif self.__training_hyperparameters._optimizer == 'sgd':
          #print('\nOptimizer: SGD')
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
        if self.__training_hyperparameters:
          model_name = self.__training_hyperparameters._model_name
        else:
          model_name = self.__test_hyperparameters._model_name
        
        if 'resnet' in model_name:
          # The order is swapped wrt VGG19 as Resnet class contains both feature extraction and classification in it
          outputs = self.__model(batch_x)
          features = self.__model.get_feature_extraction_output() 

        elif 'vgg' in model_name:
          features = self.__model.get_feature_extraction_output(batch_x)
          outputs = self.__model.get_classification_output(features)

        else:
          raise ValueError(f'Got {model_name}, but work with vgg and resnet only.')
        
        return features, outputs

    def __save_extracted_feature_vectors(self):
        """
          Salva i vettori di caratteristiche, con i relativi labels, estratti durante l'ultima epoca di training.
          I dati vengono salvati in due file .npy separati, uno per le features e uno per i labels.
          
          Il percorso del file è costruito dinamicamente in base alla data corrente (formato YYYY-MM-DD),
          al nome del modello, all'ottimizzatore utilizzato e al numero di epoche di training.

          I file vengono salvati nella directory definita da GlobalPaths.FEATURES_STEP1_CNN.
          
          filepath structure: features_step1_cnn/<YYYY-MM-DD>_<model_name>_<optimizer>_<num_epochs>_<catalog_name>_<mode>_<features/labels>.npy
        """
        if self.__training_hyperparameters:
          # Define filename based on mode:train/test
          today         = get_today_string()
          model_name    = self.__training_hyperparameters._model_name
          optimizer     = self.__training_hyperparameters._optimizer
          num_epochs    = self.__training_hyperparameters._num_epochs
          catalog_name  = self.__dataset.get_catalog_name()
          mode          = self.__training_hyperparameters._mode
          filepath_base = (
            GlobalPaths.FEATURES_STEP1_CNN / 
            f'{today}_{model_name}_{optimizer}_{num_epochs}_{catalog_name}_{mode}_'
          )
        else:
          # do stuff starting from the self.__test_hyperparameters._saved_model_name parameter
          prefix  = (self.__test_hyperparameters._saved_model_name).split("_from")[0]
          mode    = self.__test_hyperparameters._mode
          filepath_base = (
            GlobalPaths.FEATURES_STEP1_CNN /
            f'{prefix}_{mode}_'
          )

        all_features = np.concatenate(self.__extracted_features, axis=0)
        all_labels = np.concatenate(self.__extracted_labels, axis=0)
        
        np.save(filepath_base.with_name(filepath_base.name + 'features.npy'), all_features)
        np.save(filepath_base.with_name(filepath_base.name + 'labels.npy'), all_labels)

        print(f'[✓] Features and labels saved to {filepath_base}features.npy and {filepath_base}labels.npy')      

    def __save_model(self):
      """
        Salva il modello addestrato in formato .pt. Il nome del file segue uno standard basato su:
        - data corrente (YYYY-MM-DD)
        - nome del modello (es. 'resnet18')
        - ottimizzatore (es. 'adam')
        - numero di epoche
        - nome del dataset (es. 'kepler_q1q17_dr25')
        - scopo del salvataggio ('from_scratch', 'finetuned', etc.)

        Questo naming permette una chiara distinzione tra modelli addestrati da zero e quelli fine-tuned.

        Il file non verrà sovrascritto se già esistente.
      """
      today = get_today_string()
      model_name = self.__training_hyperparameters._model_name
      optimizer = self.__training_hyperparameters._optimizer
      num_epochs = self.__training_hyperparameters._num_epochs
      df_name = self.__dataset.get_catalog_name()
      purpose = self.__training_hyperparameters._purpose

      filename = f'{today}_{model_name}_{optimizer}_{num_epochs}_{df_name}_{purpose}_model.pt'

      if os.path.exists(filename):
        print(f'[WARNING] Filename: {filename}, already exists. \nThe model has not been saved to avoid overwriting.')
      else:
        torch.save(self.__model.state_dict(), GlobalPaths.TRAINED_MODELS / filename)
        print(f'[✓] Model saved in {filename}')

    def train(self):
        for epoch in tqdm(range(self.__training_hyperparameters._num_epochs), desc="Training Epochs", unit="epoch"):
          self.__model.train()    # setta il modello in modalità training
          running_loss = 0.0

          all_labels  = []
          all_outputs = []
          all_probs   = []

          # Iterate on batch during i-th training step 
          for batch_x, batch_y in self.__training_data_loader:
            # As batch_x shape is torch.Size([batch_size, 201]), we need to convert to (batch_size, in_channels=1, signal_length=201)
            # in order Resnet is able to feed-forwardly process it. For this reason, we apply the unsqueeze() method to every batch
            # before computation.  
            #NOTE DEBUG EXPERIMENTAL 
            if 'resnet' in self.__training_hyperparameters._model_name:
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
            
            # Save feature vectors during last epoch
            if epoch == self.__training_hyperparameters._num_epochs - 1:
              self.__extracted_features.append(features.detach().cpu().numpy())
              self.__extracted_labels.append(batch_y.detach().cpu().numpy())
            
            # Backpropagation
            loss.backward()
            self.__optimizer.step()
            # Update the loss function and the other metrics. Use methods from the class TrainingMetrics
            running_loss += loss.item() * batch_x.size(0)

            all_labels.extend(batch_y.detach().cpu().numpy())
            all_outputs.extend(predictions.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
          
          # Training on i-th batch has ended. Compute epoch metrics
          epoch_loss = running_loss / self.__dataset.get_training_test_set_length(split='train')
          precision = precision_score(all_labels, all_outputs, average='macro', zero_division=0)
          recall = recall_score(all_labels, all_outputs, average='macro', zero_division=0)
          f1 = f1_score(all_labels, all_outputs, average='macro', zero_division=0)
          if self.__training_hyperparameters._num_classes > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
          else:
            auc = roc_auc_score(all_labels, all_outputs)

          # Log metrics
          self.__training_metrics.log(epoch, epoch_loss, precision, recall, f1, auc)
          self.__training_metrics.print_last_classification()

        print("\nTraining completed.")        
        # Plot training metrics once training is completed. Use methods from the class TrainingMetrics
        self.__training_metrics.plot_metrics(
          output_path=GlobalPaths.TRAINING_METRICS_FEATURE_EXTRACTOR,
          model_name=self.__training_hyperparameters._model_name,
          optimizer=self.__training_hyperparameters._optimizer,
          num_epochs=self.__training_hyperparameters._num_epochs,
          df_name=self.__dataset.get_catalog_name()
          )
        
        # Concatenate and save feature vectors and labels
        self.__save_extracted_feature_vectors()

        # Save the model
        if self.__training_hyperparameters._save_model:
          self.__save_model()

    def extract_features_from_testset(self):
      # Load the model
      saved_model_name  = self.__test_hyperparameters._saved_model_name
      model_path        = GlobalPaths.TRAINED_MODELS / saved_model_name

      if not os.path.exists(model_path):
        raise ValueError(f'[ERROR] No occurrence found in "trained_models/" for {saved_model_name}.')

      self.__model.load_state_dict(torch.load(model_path, weights_only=True))

      self.__model.eval()

      # Disable gradients computation during model assessment
      with torch.no_grad():
        for batch_x, batch_y in self.__test_data_loader:

            batch_x = batch_x.unsqueeze(1)  # (batch_size, signal_length) --> (batch_size, channels, signal_length)

            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # put it on the same device where the model is
            
            # Feed-forward pass
            features, _ = self.__feed_forward_pass(batch_x)

            # Store the extracted (features,labels)
            self.__extracted_features.append(features.detach().cpu().numpy())
            self.__extracted_labels.append(batch_y.detach().cpu().numpy())
      
      # Concatenate and save feature vectors and labels
      self.__save_extracted_feature_vectors()
      
    def __del__(self):
        print('\nDestructor called for the class Model')

class FeatureExtractor:
    def __init__(self):
      """
        Costruttore della classe FeatureExtractor. Carica l'oggetto dataset e l'oggetto model
      """
      # Private attributes of the class
      self.__dataset_handler                      = self.__init_dataset()
      self.__model_hyperparameters_object         = None
      self.__model                                = None
      self.__training_test_hyperparameters_object = None
      self.__init_model()
      self.__init_training_test_hyperparameters()
    
    def __init_dataset(self):
      # 1. Initialize Dataset object with config_dataset.yaml
      with open(GlobalPaths.CONFIG / GlobalPaths.config_dataset_csv_file, 'r') as fd:
          config_fd = yaml.safe_load(fd)

      df = read_csv(GlobalPaths.CSV / config_fd['dataset_filename'])

      return Dataset(df)

    def __init_model(self):
      # 2. Initialize model architecture with config_vgg(resnet).yaml.
      with open(GlobalPaths.CONFIG / GlobalPaths.config_feature_extractor_file, 'r') as fe:
          config_fe = yaml.safe_load(fe)
      
      if config_fe['model_name'] == 'vgg':
          # load data from config_vgg.yaml
          self.__model_hyperparameters_object = InputVariablesVGG19.get_input_hyperparameters(
             GlobalPaths.CONFIG / GlobalPaths.config_vgg_file
             )
          
          # Create the model architecture
          self.__model = VGG19(
              self.__model_hyperparameters_object.get_input_size(),
              self.__model_hyperparameters_object.get_psz(),
              self.__model_hyperparameters_object.get_pst(),
              self.__model_hyperparameters_object.get_fc_layers_num(),
              self.__model_hyperparameters_object.get_fc_units(),
              self.__model_hyperparameters_object.get_fc_output_size()
              )
          print(self.__model)
      else:
          # load data from config_resnet.yaml
          self.__model_hyperparameters_object = InputVariablesResnet.get_input_hyperparameters(
             GlobalPaths.CONFIG / GlobalPaths.config_resnet_file
             )

          # Create the model
          self.__model = ResNet(
            ResidualBlock, 
            self.__model_hyperparameters_object.get_resnet_layers_num(),
            self.__model_hyperparameters_object.get_input_size(),
            self.__model_hyperparameters_object.get_fc_output_size()
            ).to(device)
          print(self.__model)
    
    def __init_training_test_hyperparameters(self):
        self.__training_test_hyperparameters_object = InputVariablesModelTraining.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_feature_extractor.yaml')

    def __feature_extraction(self):
      # Initialize Model object
      model = Model(
        self.__dataset_handler, 
        self.__model, 
        self.__model_hyperparameters_object, 
        self.__training_test_hyperparameters_object
        )
      if self.__training_test_hyperparameters_object.get_mode() == 'train':
        # Train the model
        model.train()
      else:
        # Extract features from the test set
        model.extract_features_from_testset()
      
    def main(self):
      self.__feature_extraction()

    def __del__(self):
      print('\nDestructor called for the class FeatureExtractor')

if __name__ == '__main__':
  feature_extractor = FeatureExtractor()
  feature_extractor.main()