import  sys
import  torch
import  torch.nn            as nn
import  torch.optim         as optim
import  yaml
import  os
import  re
import  numpy               as np
import  matplotlib.pyplot   as plt
from    tqdm                import tqdm
from    dataclasses         import dataclass
from    pathlib             import Path
from    typing              import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils                  import GlobalPaths, get_device, TrainingMetrics
from logger                 import log

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from dataset_mlp            import DatasetMLP

device = get_device()

@dataclass
class MLPConfig:
    mode:           str           
    input_dim:      int      
    hidden_layers:  Optional[List[int]] = None
    output_dim:     Optional[int]       = 2
    activation:     Optional[str]       = 'ReLU'
    saved_model_name: Optional[str]     = None

@dataclass
class TrainingConfig:
    batch_size:     Optional[int]   = 128
    learning_rate:  Optional[float] = None
    epochs:         Optional[int]   = None
    optimizer:      Optional[str]   = None
    loss_function:  Optional[str]   = 'MeanSquaredError'
    weight_decay:   Optional[float] = 0.0001
    momentum:       Optional[float] = 0.99

@dataclass
class DatasetConfig:
    filename_samples:       str
    filename_dispositions:  Optional[str] = None
    filename_labels:        Optional[str] = None

@dataclass
class StorageConfig:
    save_model:             Optional[bool] = False
    save_feature_vectors:   Optional[bool] = False
    plot_single:            Optional[bool] = False
    plot_per_class:         Optional[bool] = False

@dataclass
class InputVariablesMLP:
    _mlp:       MLPConfig
    _training:  TrainingConfig
    _dataset:   DatasetConfig
    _storage:   StorageConfig

    @classmethod
    def get_input_hyperparameters(cls, filename: str):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        mlp_conf        = MLPConfig(**config['mlp'])
        training_conf   = TrainingConfig(**config.get('training', {}))
        dataset_conf    = DatasetConfig(**config.get('dataset',   {}))
        storage_conf    = StorageConfig(**config.get('storage',   {}))

        return cls(
            _mlp        = mlp_conf,
            _training   = training_conf,
            _dataset    = dataset_conf,
            _storage    = storage_conf
        )

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.__mlp_hyperparameters_object = self.__init_mlp_hyperparameters()

        self.__model = self.__init_model_arch()
        self.__model.to(device)

        if self.__mlp_hyperparameters_object._mlp.mode == 'train':
            self.__training_metrics = TrainingMetrics()
            self.__optimizer = self.__init_optimizer()
            pass
        
        self.__dataset = self.__init_dataset()

        self.__loss_fn = self.__init_loss()
        
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
            self.__training_metrics.print_last_regression()
        #end training
    
    def __save_projected_feature_vectors(self, filename_features:str, filename_labels:str):
        """
            Salva i vettori di caratteristiche proiettati dal MLP nello spazio 2D durante l'ultima epoca di training
        """
        # Define output path
        filepath_features_base = (
            GlobalPaths.FEATURES_STEP2_MLP /
            f'{filename_features}.npy'
        )
        filepath_labels_base = (
            GlobalPaths.FEATURES_STEP2_MLP /
            f'{filename_labels}.npy'
        )
        if os.path.exists(filepath_features_base):
            raise ValueError(f'[WARNING] Filename: {filename_features}, already exists. \n Features not saved!')

        all_features = np.concatenate(self.__projected_features, axis=0)
        all_labels = self.__dataset.get_dispositions()
        if len(all_features) != len(all_labels):
            raise ValueError(f'[WARNING] In saving <features_2d,labels>, mismatch in length. len(all_features)={len(all_features)}, len(all_labels)={len(all_labels)}')

        np.save(filepath_features_base.with_name(filepath_features_base.name), all_features)
        np.save(filepath_labels_base.with_name(filepath_labels_base.name), all_labels)

        print(f'[✓] <Features,Labels> projected by MLP saved to {filepath_features_base} and {filepath_labels_base}')
  
    def __save_model(self, filename:str):
        filepath_base = (
            GlobalPaths.TRAINED_MODELS /
            filename
        )
        if os.path.exists(filepath_base):
            log.warning(f'[WARNING] Filename: {filename}, already exists. \nThe model has not been saved to avoid overwriting.')
        else:
            torch.save(self.__model.state_dict(), filepath_base)
            log.info(f'[✓] Model saved in {filepath_base}')

    def __plot_mlp_representation(self, filename:str, alpha=0.4):
        fontsize    = 24
        resolution  = 1200
        labels      = self.__dataset.get_dispositions()
        projection  = np.vstack(self.__projected_features)    # To avoid the error: TypeError: list indices must be integers or slices, not tuple
        
        plt.figure(figsize=(10, 8))

        color_map = {
            0: "#3d014b",   # viola
            1: "#21918c",   # verde acqua
            2: "#fde724"    # giallo
        }
        class_map = {
            0: "EB",
            1: "PC",
            2: "J"
        }

        # Plotta una classe alla volta per controllare colore e legenda
        for cls, color in color_map.items():
            mask = labels == cls
            plt.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=color,
                alpha=alpha,
                label=class_map[cls],
            )

        plt.xlabel('MLP Dimension 1', fontsize=fontsize)
        plt.ylabel('MLP Dimension 2', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylim(-2.5, 2.5)
        plt.xlim(-2.5, 2.5)

        # Aggiungi legenda
        plt.legend(title="Classes", fontsize=fontsize-4, title_fontsize=fontsize-2)

        filepath_base = (
            GlobalPaths.OUTPUT_FILES / GlobalPaths.PLOT_MLP / 
            f'{filename}.png'
          )
        plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
        plt.close()

    def __plot_mlp_representation_single_class(self, filename, target_class, alpha=0.6):
        """
            Plotta la proiezione MLP solo per gli elementi appartenenti a una classe specifica.

            Parameters
            ----------
            filename : str
                Nome base del file di output (senza estensione).
            target_class : int
                Classe da plottare (0, 1 o 2).
        """
        fontsize    = 24
        resolution  = 1200

        # Recupera etichette e features proiettate
        labels      = self.__dataset.get_dispositions()
        projection  = np.vstack(self.__projected_features)  # shape (N, 2)

        # Filtra solo la classe desiderata
        mask = labels == target_class
        filtered_projection = projection[mask]
        filtered_labels = labels[mask]

        if filtered_projection.size == 0:
            raise ValueError(f"[!] Nessun campione trovato per la classe {target_class}")

        plt.figure(figsize=(10, 8))

        color_map = {
            0: "#3d014b",   # viola
            1: "#21918c",   # verde acqua
            2: "#fde724"    # giallo
        }
        class_map = {
            0: "EB",
            1: "PC",
            2: "J"
        }

        # Colore corrispondente alla classe scelta
        class_color = color_map.get(target_class, "#000000")  # default nero se non trovato
        class_label = class_map.get(target_class, "UNK")

        scatter = plt.scatter(
            filtered_projection[:, 0],
            filtered_projection[:, 1],
            c=class_color,
            alpha=alpha,
            label=f"{class_label}"
        )

        #plt.colorbar(scatter, label=f"Class {target_class}")
        plt.xlabel("MLP Dimension 1", fontsize=fontsize)
        plt.ylabel("MLP Dimension 2", fontsize=fontsize)
        plt.ylim(-2.5, 2.5)
        plt.xlim(-2.5, 2.5)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize-4)

        filepath_base = (
            GlobalPaths.OUTPUT_FILES / GlobalPaths.PLOT_MLP / f"{filename}_class{target_class}.png"
        )

        plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
        plt.close()
        log.debug(f"[✓] Plot saved to: {filepath_base}")

    def __project_features_from_testset(self):
        # Load the model
        saved_model_name    = self.__mlp_hyperparameters_object._mlp.saved_model_name
        model_path          = GlobalPaths.TRAINED_MODELS / saved_model_name

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

    def __build_output_filenames(self):
        mode   = self.__mlp_hyperparameters_object._mlp.mode
        
        # Build the prefix of the filename containing the projected features by MLP
        prefix = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split(f'_{mode}')[0]}"
        
        if self.__mlp_hyperparameters_object._training.epochs is not None:
            epochs = self.__mlp_hyperparameters_object._training.epochs
        else:
            # Get the number of epochs from saved model name
            result = re.search('mlp_(.*).pt', self.__mlp_hyperparameters_object._mlp.saved_model_name)
            epochs = result.group(1)
        
        filename_features   = f'{prefix}_{mode}_features_2d_mlp_{epochs}'
        filename_labels     = f'{prefix}_{mode}_labels_2d_mlp_{epochs}'

        if mode == 'train':
            filename_model  = f'{prefix}_from_scratch_mlp_{epochs}.pt'
            filename_loss   = f'{prefix}_loss_{epochs}.png'
            return filename_features, filename_labels, filename_model, filename_loss
        else:
            return filename_features, filename_labels

    def main(self):

        print(self.__model)

        mode   = self.__mlp_hyperparameters_object._mlp.mode
        """
        # Build the prefix of the filename containing the projected features by MLP
        prefix = f"{(self.__mlp_hyperparameters_object._dataset.filename_samples).split(f'_{mode}')[0]}"
        if self.__mlp_hyperparameters_object._training.epochs is not None:
            epochs = self.__mlp_hyperparameters_object._training.epochs
        else:
            # Get the number of epochs from saved model name
            result = re.search('mlp_(.*).pt', self.__mlp_hyperparameters_object._mlp.saved_model_name)
            epochs = result.group(1)
        
        filename_features   = f'{prefix}_{mode}_features_2d_mlp_{epochs}'
        filename_labels     = f'{prefix}_{mode}_labels_2d_mlp_{epochs}'
        """

        if mode == 'train':
            self.__train()

            # Training completed, define output filenames
            filename_features, filename_labels, filename_model, filename_loss = self.__build_output_filenames()
            
            self.__training_metrics.plot_loss(
                output_path = str(GlobalPaths.OUTPUT_FILES / GlobalPaths.PLOT_MLP),
                filename    = filename_loss
                )                                                               # Plot loss after training
            
            if self.__mlp_hyperparameters_object._storage.save_model == True:
                self.__save_model(filename_model)                               # Save the model
        else:
            filename_features, filename_labels = self.__build_output_filenames()
            self.__project_features_from_testset()

        # This code is executed in both modes train and test
        if self.__mlp_hyperparameters_object._storage.plot_single == True:
            self.__plot_mlp_representation(filename_features)                   # Plot MLP representation      
        
        if self.__mlp_hyperparameters_object._storage.plot_per_class == True:
            for i in range(3):
                self.__plot_mlp_representation_single_class(filename_features, i)
        
        if self.__mlp_hyperparameters_object._storage.save_feature_vectors == True:
            self.__save_projected_feature_vectors(filename_features, filename_labels)   # Concatenate and save feature vectors and labels


if __name__ == "__main__":
    mlp = MLP()
    mlp.main()