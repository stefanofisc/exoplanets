import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

### da rivedere
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split 
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
    filename_dispositions: str
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

        for epoch in tqdm(range(num_epochs), desc="[MLP] Training Epochs", unit="epoch"):
            self.__model.train()    # set the model in training mode
            running_loss = 0.0

            all_labels = []
            all_outputs = []
            all_probs = []

            for batch_x, batch_y in training_set_loader:
                # Check if batch_x shape is [batch_size, input_dim]. the use of unsqueeze() is unnecessary
                assert batch_x.ndim == 2, f"[ERROR] Expected 2D input, got {batch_x.ndim}D"
                assert batch_x.shape[1] == self.__mlp_hyperparameters_object._mlp.input_dim, (
                f"[ERROR] Expected input dim {self.__mlp_hyperparameters_object._mlp.input_dim}, got {batch_x.shape[1]}"
                )

                batch_y = batch_y.float()

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                self.__optimizer.zero_grad()

                outputs = self.__forward(batch_x)   # outputs.shape = (batch_size, output_dim=2)

                loss = self.__loss_fn(outputs, batch_y)
                loss.backward()                     # Backpropagation
                
                self.__optimizer.step()

                running_loss += loss.item()
            
            epoch_loss = running_loss / len(training_set_loader)
            self.__training_metrics.log(epoch, epoch_loss)
            self.__training_metrics.print_last()
        
        # Plot loss after training
        plot_filename = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split('_train')[0]}_loss.png"
        self.__training_metrics.plot_loss(
            output_path=str(GlobalPaths.OUTPUT_FILES / 'mlp_training_metrics'),
            filename = plot_filename
            )
        
    def main(self):
        print(self.__model)
        print(self.__optimizer)
        print(self.__loss_fn)
        self.__train()

    
    """
    def __get_samples_labels(self):
        # Get dataset (split 100%)
        samples = np.load(self.dataset_config["samples"])
        labels = np.load(self.dataset_config["labels"])
        dispostions = np.load(self.dataset_config["dispositions"])
        
        self._get_dataset_statistics(samples, labels)
        # Normalize data to zero mean and unit variance
        epsilon = 1e-8  # offset to improve numerical stability. This prevents division by zero for features with zero std
        normalized_samples = (samples - samples.mean()) / ( samples.std() + epsilon )
        normalized_labels = (labels - labels.mean()) / ( labels.std() + epsilon )
        self._get_dataset_statistics(normalized_samples, normalized_labels)

        # split dataset into train-test
        # Qui non uso stratify per bilanciare le classi visto che non è un problema di classificazione ma di encoding supervisionato
        x_train, x_test, y_train, y_test, d_train, d_test = train_test_split(normalized_samples, normalized_labels, dispostions, test_size=0.2, random_state=42)

        print("Dataset size:\n")
        print(f"x_train: {x_train.shape}; y_train: {y_train.shape}; d_train: {d_train.shape}")
        print(f"x_test: {x_test.shape}; y_test: {y_test.shape}; d_test: {d_test.shape}")

        # Convert numpy.ndarray data into torch.Tensor
        normalized_x_train = torch.tensor(x_train, dtype=torch.float32) # Training
        normalized_y_train = torch.tensor(y_train, dtype=torch.float32) # Training
        normalized_x_test = torch.tensor(x_test, dtype=torch.float32) # Test
        normalized_y_test = torch.tensor(y_test, dtype=torch.float32) # Test

        # Create DataLoader object for training/testing the model
        training_set = TensorDataset(normalized_x_train, normalized_y_train)
        test_set = TensorDataset(normalized_x_test, normalized_y_test)

        self.training_set_loader = DataLoader(training_set, batch_size=self.train_config["batch_size"], shuffle=False)
        self.test_set_loader = DataLoader(test_set, batch_size=self.train_config["batch_size"], shuffle=False)
        self.training_set_loader_dispositions = d_train     #numpy.ndarray
        self.test_set_loader_dispositions = d_test
        #return training_set_loader, test_set_loader
    """

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