import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import GlobalPaths, get_device, TrainingMetrics

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from dataset import DatasetMLP

device = get_device()

@dataclass
class MLPConfig:
    mode: str           
    input_dim: int      
    hidden_layers: Optional[List[int]] = None
    output_dim: Optional[int] = 2
    activation: Optional[str] = 'ReLU'
    saved_model_name: Optional[str] = None

@dataclass
class TrainingConfig:
    batch_size: Optional[int] = 128
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss_function: Optional[str] = 'MeanSquaredError'
    weight_decay: Optional[float] = 0.0001
    momentum: Optional[float] = 0.99

@dataclass
class DatasetConfig:
    filename_samples: str
    filename_dispositions: Optional[str] = None
    filename_labels: Optional[str] = None

@dataclass
class InputVariablesMLP:
    _mlp: MLPConfig
    _training: TrainingConfig
    _dataset: DatasetConfig

    @classmethod
    def get_input_hyperparameters(cls, filename: str):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        mlp_conf = MLPConfig(**config['mlp'])
        training_conf = TrainingConfig(**config.get('training', {}))
        dataset_conf = DatasetConfig(**config.get('dataset', {}))

        return cls(
            _mlp=mlp_conf,
            _training=training_conf,
            _dataset=dataset_conf
        )

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.__mlp_hyperparameters_object = self.__init_mlp_hyperparameters()   #ok

        self.__model = self.__init_model_arch()     #ok
        self.__model.to(device)                     #ok

        if self.__mlp_hyperparameters_object._mlp.mode == 'train':
            self.__training_metrics = TrainingMetrics()
            self.__optimizer = self.__init_optimizer()
            pass
        
        self.__dataset = self.__init_dataset()      #ok

        self.__loss_fn = self.__init_loss()         #ok
        
        self.__projected_features = []

    def __init_model_arch(self):
        # Input variables
        layers = []
        input_dim = self.__mlp_hyperparameters_object._mlp.input_dim
        activation = self.__mlp_hyperparameters_object._mlp.activation
        hidden_layers = self.__mlp_hyperparameters_object._mlp.hidden_layers
        output_dim = self.__mlp_hyperparameters_object._mlp.output_dim

        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())  # Dynamically fetch activation function
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # Output layer
        
        return nn.Sequential(*layers)
    
    def __init_mlp_hyperparameters(self):
        return InputVariablesMLP.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_mlp.yaml')

    def __init_dataset(self):
        """
            Init the dataset object
        """
        return DatasetMLP(
            self.__mlp_hyperparameters_object._mlp.mode,
            self.__mlp_hyperparameters_object._dataset, 
            self.__mlp_hyperparameters_object._training
            )

    def __init_loss(self):
        """
            Init the loss function
        """
        if self.__mlp_hyperparameters_object._mlp.mode == 'train':
            loss_function = self.__mlp_hyperparameters_object._training.loss_function
            if loss_function == 'MeanSquaredError':
                return nn.MSELoss()
            else:
                raise ValueError(f'[ERROR] Got {loss_function}, but other loss functions still not available. Please use MeanSquaredError')
        else:
            #NOTE. This block of code might change in future implementations whether new loss function would be provided
            return nn.MSELoss()

    def __init_optimizer(self):
        """
            Init the optimization algorithm.
            #TODO. Creare classe da cui 'mlp_class/MLP' e 'feature_extractor/Model' ereditano metodi in comune,
                    che per adesso è solo __init_optimizer
        """
        optimizer = self.__mlp_hyperparameters_object._training.optimizer
        learning_rate = self.__mlp_hyperparameters_object._training.learning_rate

        if optimizer == 'adam':
            return optim.Adam(self.__model.parameters(), lr=learning_rate)
        
        elif optimizer == 'sgd':
            weight_decay = self.__mlp_hyperparameters_object._training.weight_decay
            momentum = self.__mlp_hyperparameters_object._training.momentum
            return optim.SGD(
                self.__model.parameters(),
                lr = learning_rate,
                weight_decay = weight_decay,
                momentum = momentum
            )
        else:
          raise ValueError(f'Got {optimizer}, but work with Adam and Stochastic Gradient Descent optimizers only.\n Please set adam or sgd to train the model.')

    def __forward(self, x):
        return self.__model(x)
    
    def __train(self):
        """
            Train the MLP
        """
        num_epochs = self.__mlp_hyperparameters_object._training.epochs
        batch_size = self.__mlp_hyperparameters_object._training.batch_size
        training_set_loader = self.__dataset.get_training_data_loader()
        self.__dataset.set_dispositions(training_set_loader.dataset.tensors[2]) # prendi le dispositions shuffled

        for epoch in tqdm(range(num_epochs), desc="[MLP] Training Epochs", unit="epoch"):
            self.__model.train()    # set the model in training mode
            running_loss = 0.0

            for batch_x, batch_y, _ in training_set_loader:
                # In alternativa:
                #for batch in training_set_loader: batch_x, batch_y = batch[:2]  # estrai solo X e y
                # Check if batch_x shape is [batch_size, input_dim]. the use of unsqueeze() is unnecessary
                assert batch_x.ndim == 2, f"[ERROR] Expected 2D input, got {batch_x.ndim}D"
                assert batch_x.shape[1] == self.__mlp_hyperparameters_object._mlp.input_dim, (
                f"[ERROR] Expected input dim {self.__mlp_hyperparameters_object._mlp.input_dim}, got {batch_x.shape[1]}"
                )

                batch_y = batch_y.float()

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                self.__optimizer.zero_grad()

                outputs = self.__forward(batch_x)   # outputs.shape = (batch_size, output_dim=2)

                if epoch == num_epochs - 1:
                    self.__projected_features.append(outputs.detach().cpu().numpy())        

                loss = self.__loss_fn(outputs, batch_y)
                loss.backward()                     # Backpropagation
                
                self.__optimizer.step()

                running_loss += loss.item()
            
            epoch_loss = running_loss / len(training_set_loader)
            self.__training_metrics.log_loss(epoch, epoch_loss)
            self.__training_metrics.print_last()
        #end training
    
    def __save_projected_feature_vectors(self, filename:str):
        """
            Salva i vettori di caratteristiche proiettati dal MLP nello spazio 2D durante l'ultima epoca di training
        """
        # Define output path
        filepath_base = (
            GlobalPaths.FEATURES_STEP2_MLP /
            f'{filename}.npy'
        )
        if os.path.exists(filepath_base):
            raise ValueError(f'[WARNING] Filename: {filename}, already exists. \n Features not saved!')

        all_features = np.concatenate(self.__projected_features, axis=0)
        np.save(filepath_base.with_name(filepath_base.name), all_features)

        print(f'[✓] Features projected by MLP saved to {filepath_base}')      
    
    def __save_model(self, filename:str):
        filepath_base = (
            GlobalPaths.TRAINED_MODELS /
            filename
        )
        if os.path.exists(filepath_base):
            print(f'[WARNING] Filename: {filename}, already exists. \nThe model has not been saved to avoid overwriting.')
        else:
            torch.save(self.__model.state_dict(), filepath_base)
            print(f'[✓] Model saved in {filepath_base}')

    def __plot_mlp_representation(self, filename:str):
        fontsize = 20
        resolution = 1200
        labels = self.__dataset.get_dispositions()
        projection = np.vstack(self.__projected_features)    # To avoid the error: TypeError: list indices must be integers or slices, not tuple
        
        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class Labels')

        plt.xlabel('MLP Dimension 1', fontsize=fontsize)
        plt.ylabel('MLP Dimension 2', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        #NOTE. plot_mlp, plot_tsne, plot_cnn_training_metrics, posso inserirli tutti in utils.py
        filepath_base = (
            GlobalPaths.OUTPUT_FILES / 'plot_mlp' / 
            f'{filename}.png'
          )

        plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
        plt.close()

    def __project_features_from_testset(self):
        # Load the model
        saved_model_name = self.__mlp_hyperparameters_object._mlp.saved_model_name
        model_path = GlobalPaths.TRAINED_MODELS / saved_model_name

        if not os.path.exists(model_path):
            raise ValueError(f'[ERROR] No occurrence found in "trained_models/" for {saved_model_name}.')

        self.__model.load_state_dict(torch.load(model_path, weights_only=True))

        self.__model.eval()

        with torch.no_grad():
            for batch in self.__dataset.get_test_data_loader():
                # Qui stiamo iterando su un DataLoader che è stato costruito a partire da un TensorDataset 
                # che contiene un solo tensore, anziché due (x_test,y_test). Per cui la seguente linea
                # di codice varia leggermente. Ogni elemento di questo DataLoader è una tupla di un solo elemento: (x,).
                batch_x = batch[0].to(device)

                outputs = self.__forward(batch_x)

                # Store the projected features into a <class 'list'>, where each element is <class 'numpy.ndarray'>
                self.__projected_features.append(outputs.detach().cpu().numpy())


    def main(self):

        print(self.__model)

        mode = self.__mlp_hyperparameters_object._mlp.mode

        # Prefix of the filename containing the projected features by MLP
        prefix = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split(f'_{mode}')[0]}"
        filename_features = f'{prefix}_{mode}_features_2d_mlp'

        if mode == 'train':
            self.__train()
            
            # Training completed. Define the filenames for saving plots, features and model
            #NOTE. Remove these two lines if new prefix works. prefix = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split('_train')[0]}"
            # filename_features = f'{prefix}_train_features_2d_mlp'
            filename_model = f'{prefix}_from_scratch_mlp.pt'
            
            self.__training_metrics.plot_loss(
                output_path=str(GlobalPaths.OUTPUT_FILES / 'plot_mlp'),
                filename = f'{prefix}_loss.png'
                )                                                       # Plot loss after training
            
            #NOTE. Remove
            #NOTE.self.__plot_mlp_representation(filename_features)           # Plot MLP representation      
            #NOTE.self.__save_projected_feature_vectors(filename_features)    # Concatenate and save feature vectors and labels
            self.__save_model(filename_model)                           # Save the model
        else:
            self.__project_features_from_testset()

        # Code executed in both modes train and test
        self.__plot_mlp_representation(filename_features)           # Plot MLP representation      
        self.__save_projected_feature_vectors(filename_features)    # Concatenate and save feature vectors and labels

    
    """
    def __save_model_output(self, output, mode='train'):
        features_2d_array = np.vstack(output)
        print(f"features shape: {features_2d_array.shape}")

        if mode=='train':
            filename_output = self.dataset_config["model_output_filename"]                  # coordinate 2D delle features prodotte dal MLP
            filename_dispositions = self.dataset_config["training_dispositions_filename"]   # etichetta del punto: {AFP, NTP, ecc...}
            features_2d_disp = np.vstack(self.training_set_loader_dispositions)
        else:
            filename_output = self.dataset_config["model_output_testset_filename"]
            filename_dispositions = self.dataset_config["test_dispositions_filename"]
            features_2d_disp = np.vstack(self.test_set_loader_dispositions)
        
        np.save(filename_output, features_2d_array)
        np.save(filename_dispositions, features_2d_disp)
    """

    """
    def evaluate(self, device):
        self.to(device)
        
        # Initialize metrics
        total_loss = 0
        predictions = []
        actual = []
        
        # Define loss function (same as used in training)
        loss_fn = nn.MSELoss()
        
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in self.test_set_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(inputs)
                
                # Calculate loss
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # Store predictions and actual values
                predictions.append(outputs.cpu().numpy())
                actual.append(labels.cpu().numpy())

                self.__test_set_output = predictions
        self.__save_model_output(self.__test_set_output, 'test')

        # Calculate average loss
        avg_loss = total_loss / len(self.test_set_loader)
        
        # Concatenate batches
        all_predictions = np.concatenate(predictions)
        all_actual = np.concatenate(actual)
        
        # Calculate metrics
        mse = np.mean((all_predictions - all_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_actual))
        
        # Calculate R² (coefficient of determination)
        ss_total = np.sum((all_actual - np.mean(all_actual)) ** 2)
        ss_residual = np.sum((all_actual - all_predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Print evaluation results
        print("\nTest Set Evaluation:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r_squared:.4f}")
        
        return {
            "loss": avg_loss,
            "mse": mse,
            "rmse": rmse, 
            "mae": mae,
            "r_squared": r_squared,
            "predictions": all_predictions,
            "actual": all_actual
        }
    """


if __name__ == "__main__":
    mlp = MLP()
    mlp.main()