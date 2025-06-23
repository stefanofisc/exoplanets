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

### da rivedere
#from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split 
### end

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import GlobalPaths, get_device, TrainingMetrics

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from dataset import DatasetMLP


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = get_device()

@dataclass
class MLPConfig:
    input_dim: int
    hidden_layers: List[int]
    output_dim: int
    activation: str

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    optimizer: str
    loss_function: str
    weight_decay: Optional[float] = 0.0001
    momentum: Optional[float] = 0.99

@dataclass
class DatasetConfig:
    filename_samples: str
    filename_dispositions: Optional[str] = None #NOTE Remove?
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
        training_conf = TrainingConfig(**config['training'])
        dataset_conf = DatasetConfig(**config['dataset'])

        return cls(
            _mlp=mlp_conf,
            _training=training_conf,
            _dataset=dataset_conf
        )

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.__mlp_hyperparameters_object = self.__init_mlp_hyperparameters()
        self.__training_metrics = TrainingMetrics()

        self.__model = self.__init_model_arch()
        self.__model.to(device)
        
        self.__dataset = self.__init_dataset()

        self.__loss_fn = self.__init_loss()
        
        self.__optimizer = self.__init_optimizer()

        self.__projected_features = []
        #self.__projected_features_dispositions = []

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

        """
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.training_set_loader = None
        self.training_set_loader_dispositions = None    #numpy.ndarray
        self.test_set_loader = None
        self.test_set_loader_dispositions = None
        self.__model_output = []
        self.__test_set_output = []
        """
    
    def __init_mlp_hyperparameters(self):
        return InputVariablesMLP.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_mlp.yaml')
        #print(self.__mlp_hyperparameters_object._mlp.hidden_layers)
        #print(self.__mlp_hyperparameters_object._training.optimizer)

    def __init_dataset(self):
        """
            Init the dataset object
        """
        return DatasetMLP(
            self.__mlp_hyperparameters_object._dataset, 
            self.__mlp_hyperparameters_object._training
            )

    def __init_loss(self):
        """
            Init the loss function
        """
        loss_function = self.__mlp_hyperparameters_object._training.loss_function
        if loss_function == 'MeanSquaredError':
            return nn.MSELoss()
        else:
            raise ValueError(f'[ERROR] Got {loss_function}, but other loss functions still not available. Please use MeanSquaredError')
    
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

    def main(self):
        print(self.__model)
        self.__train()
        
        # Training completed. Define the filenames for saving plots, features and model
        prefix = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split('_train')[0]}"
        filename_features = f'{prefix}_train_features_2d_mlp'
        filename_model = f'{prefix}_from_scratch_mlp.pt'
        
        self.__training_metrics.plot_loss(
            output_path=str(GlobalPaths.OUTPUT_FILES / 'plot_mlp'),
            filename = f'{prefix}_loss.png'
            )                                                       # Plot loss after training
        
        self.__plot_mlp_representation(filename_features)           # Plot MLP representation      
        self.__save_projected_feature_vectors(filename_features)    # Concatenate and save feature vectors and labels
        self.__save_model(filename_model)                           # Save the model


    
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

    #NOTE TB deleted
    def train_model(self, loss_fn, optimizer, device, epochs):
        self.to(device)
        # get training data
        self.__get_samples_labels()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            #############################################################
            for inputs, labels in self.training_set_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()   # Initialize gradients
                output = self(inputs)   # Feed-forward pass
                #labels = labels.squeeze() # Remove any additional dimensionality from the data
                
                loss = loss_fn(output, labels)
                loss.backward()         # Backpropagation
                optimizer.step()
                total_loss += loss.item() #* inputs.size(0)

                if epoch == epochs - 1:
                    self.__model_output.append(output.detach().cpu().numpy())
            #############################################################
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(self.training_set_loader):.4f}")
        # Save model output
        self.__save_model_output(self.__model_output, 'train')


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

"""
# Main function to run the pipeline
def main():
    # Load configuration
    config = load_config()
    
    # Unpack configuration
    mlp_config = config["mlp"]
    train_config = config["training"]
    dataset_config = config["dataset"]

    #input_dim = eval(str(mlp_config["input_dim"]))
    input_dim = mlp_config["input_dim"]
    hidden_layers = mlp_config["hidden_layers"]
    output_dim = mlp_config["output_dim"]
    activation = mlp_config["activation"]

    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    epochs = train_config["epochs"]
    optimizer_name = train_config["optimizer"]
    loss_function_name = train_config["loss_function"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model summary
    model = MLP(input_dim, hidden_layers, output_dim, activation, train_config, dataset_config)
    print("Network Architecture:\n")
    print(model)
    

    # Initialize loss function, and optimizer
    if loss_function_name == "MeanSquaredError":
        loss_function = nn.MSELoss()
    else:
        loss_function = getattr(nn, loss_function_name)()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Train the model
    model.train_model(loss_function, optimizer, device, epochs)
    model.evaluate(device)
"""

if __name__ == "__main__":
    mlp = MLP()
    mlp.main()