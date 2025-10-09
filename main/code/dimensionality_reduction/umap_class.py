import  yaml
import  sys
from    pathlib                 import Path
from    dataclasses             import dataclass
from    pandas                  import read_csv
import  torch.nn                as nn
import  numpy                   as np
import  matplotlib.pyplot       as plt
from    typing                  import Optional
from    umap.parametric_umap    import ParametricUMAP
#from    umap                    import UMAP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils                   import GlobalPaths, get_device, get_today_string
from    logger                  import log

sys.path.insert(1, str(GlobalPaths.DATASET))
from    dataset                 import Dataset

sys.path.insert(2, str(GlobalPaths.FEATURE_EXTRACTION))
from    resnet.resnet_class         import InputVariablesResnet
from    resnet.resnet_tf_class      import ResNetTF
from    vgg.vgg_class               import VGG19, InputVariablesVGG19       #NOTE To be modified

from sklearn.svm import SVC

device = get_device()
print(f"Executing UMAP on {device}")

@dataclass
class InputVariablesUMAP:
    # Define the encoder
    _model_name:    str
    # UMAP parameters
    _n_components:  int
    _n_neighbors:   int
    _min_dist:      float
    _metric:        str
    _random_state:  Optional[int]
    _n_epochs:      Optional[int]

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _model_name    = config.get('model_name', 'resnet'),
            _n_components  = config.get('n_components', 3),
            _n_neighbors   = config.get('n_neighbors', 15),
            _min_dist      = config.get('min_dist', 0.1),
            _metric        = config.get('metric', 'euclidean'),
            _random_state  = config.get('random_state', 42),
            _n_epochs      = config.get('n_epochs', 200)
            )

class myUMAP:
    def __init__(self):
        """
            Costruttore della classe myUMAP. Carica l'oggetto dataset e l'oggetto model che fungerà da encoder.
            In questo costruttore vengono definiti ed inizializzati 3 oggetti:
                (1) Dataset, (2) VGG19 o Resnet, (3) UMAP
        """
        # Objects initialization
        self.__dataset_handler              = self.__init_dataset() # dataset.Dataset
        self.__model_hyperparameters_object = None                  # dati di input della CNN presi da config_vgg o config_resnet
        self.__model                        = self.__init_model()   # VGG19 or Resnet
        self.__umap_hyperparameters_object  = self.__init_umap_hyperparameters()

        self.__encoder_instance             = self.__get_feature_extractor()
        self.__projected_features           = None
        self.__extracted_labels             = None # Loaded in project_data()
        self.__mapper                       = None # initialized in project_data()

        log.info('UMAP class initialized.')

    def __init_dataset(self):
        # 1. Initialize Dataset object with config_dataset.yaml
        with open(GlobalPaths.CONFIG / GlobalPaths.config_dataset_csv_file, 'r') as fd:
            config_fd = yaml.safe_load(fd)

        df = read_csv(GlobalPaths.CSV / config_fd['dataset_filename'])

        return Dataset(df)
    
    def __init_model(self):
        # Initialize model architecture with config_vgg(resnet).yaml.
        with open(GlobalPaths.CONFIG / GlobalPaths.config_umap_file, 'r') as fu:
            config_fu = yaml.safe_load(fu)
        
        model_name = config_fu['model_name']
        
        if model_name == 'vgg':
            # load data from config_vgg.yaml
            self.__model_hyperparameters_object = InputVariablesVGG19.get_input_hyperparameters(
                GlobalPaths.CONFIG / GlobalPaths.config_vgg_file
                )
            # Create the model architecture
            model = VGG19(
                self.__model_hyperparameters_object.get_input_size(),
                self.__model_hyperparameters_object.get_psz(),
                self.__model_hyperparameters_object.get_pst(),
                self.__model_hyperparameters_object.get_fc_layers_num(),
                self.__model_hyperparameters_object.get_fc_units(),
                self.__model_hyperparameters_object.get_fc_output_size()
                ).to(device)
            
            log.info('Initialized VGG19 model.')
            return model
        
        elif 'resnet' in model_name:
            # load data from config_resnet.yaml
            self.__model_hyperparameters_object = InputVariablesResnet.get_input_hyperparameters(
                GlobalPaths.CONFIG / GlobalPaths.config_resnet_file
                )
            # Create the model
            model = ResNetTF(
                self.__model_hyperparameters_object.get_resnet_layers_num(),
                self.__model_hyperparameters_object.get_input_size(),
                self.__model_hyperparameters_object.get_fc_output_size()
            )
            log.info(f'Initialized {model_name} model.')
            return model
        
        else:
            raise ValueError(f'Model {model_name} not supported for UMAP encoder.')

    def __init_umap_hyperparameters(self):
        return InputVariablesUMAP.get_input_hyperparameters(GlobalPaths.CONFIG / GlobalPaths.config_umap_file)

    def __get_feature_extractor(self):
        """
            Extracts the feature extraction block (encoder) from the CNN model.
            For ResNet (TensorFlow version), it calls get_feature_extraction_model().
            For VGG, the implementation will be added in the future.
        """
        full_model   = self.__model
        model_name   = self.__umap_hyperparameters_object._model_name

        if 'resnet' in model_name:
            # Ensure the model has been built at least once
            dummy_input = np.random.randn(1, self.__model_hyperparameters_object.get_input_size()).astype(np.float32)
            _ = full_model(dummy_input)  # Build graph

            # Extract the encoder submodel (feature extractor)
            self.__encoder_instance = full_model.get_feature_extraction_model()

            log.info("[✓] Feature extractor (ResNetTF) successfully created for Parametric UMAP.")

        elif model_name == 'vgg':
            # Future implementation: TensorFlow-based VGG19 encoder
            log.warning("[!] VGG encoder extraction not implemented yet for TensorFlow version.")
            self.__encoder_instance = None

        else:
            raise ValueError(f"Unsupported model '{model_name}' for UMAP encoder extraction.")

    def project_data(self):
        """
            Addestra l'istanza di ParametricUMAP sull'encoder Resnet/VGG e proietta i dati nel sottospazio.
        """
        # 1. Loading training data
        X_train_full, Y_train_full, _, _ = self.__dataset_handler.get_training_test_samples()
        
        # 2. Converting into numpy.ndarray
        X_train_np = X_train_full.cpu().numpy()
        Y_train_np = Y_train_full.cpu().numpy()
            
        # 3. Initializing ParametricUMAP
        self.__mapper = ParametricUMAP(
            encoder     =   self.__encoder_instance,
            n_components=   self.__umap_hyperparameters_object._n_components,
            n_neighbors =   self.__umap_hyperparameters_object._n_neighbors,
            min_dist    =   self.__umap_hyperparameters_object._min_dist,
            metric      =   self.__umap_hyperparameters_object._metric,
            random_state=   self.__umap_hyperparameters_object._random_state,
            n_epochs    =   self.__umap_hyperparameters_object._n_epochs,
            parametric_reconstruction = True
        )
        
        log.info(f"Projecting with Parametric UMAP ({self.__umap_hyperparameters_object._n_components}D)...")
        
        # 4. Fit the model
        self.__mapper.fit(X_train_np, Y_train_np)

        self.__projected_features = self.__mapper.transform(X_train_np)
        self.__extracted_labels   = Y_train_np

        log.info(f"[✓] Parametric UMAP proiezione completata. Shape: {self.__projected_features.shape}")


    def classify_data(self):
        _, _, X_test_full, Y_test_full = self.__dataset_handler.get_training_test_samples()

        X_test_np = X_test_full.cpu().numpy()
        Y_test_np = Y_test_full.cpu().numpy()

        # Train the support vector machine
        svc = SVC().fit(self.__projected_features, self.__extracted_labels)

        print('train: ', svc.score(self.__projected_features, self.__extracted_labels))
        print('test: ', svc.score(self.__mapper.transform(X_test_np), Y_test_np))
        
    def save_projected_data(self):
        pass

    def plot_projection(self):
        """
            Genera un plot 2D o 3D (statico) dei dati proiettati, colorati per classe.
        """
        if self.__projected_features is None:
            log.error("Proiezione non eseguita. Chiamare project_data() prima di plottare.")
            return
            
        n_components = self.__umap_hyperparameters_object._n_components
        model_name   = self.__umap_hyperparameters_object._model_name
        labels       = self.__extracted_labels
        
        fig = plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                self.__projected_features[:, 0], 
                self.__projected_features[:, 1], 
                c=labels, 
                cmap='viridis', 
                alpha=0.7
            )
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_title(f'Parametric UMAP ({model_name}) Projection (2D)')
            
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                self.__projected_features[:, 0],
                self.__projected_features[:, 1],
                self.__projected_features[:, 2],
                c=labels,
                cmap='viridis',
                edgecolors='k',
                alpha=0.7
            )
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_zlabel('UMAP Dimension 3')
            ax.set_title(f'Parametric UMAP ({model_name}) Projection (3D)')
            
        else:
            log.error(f"Visualizzazione supportata solo per n_components = 2 o 3, ma è {n_components}")
            plt.close(fig)
            return

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Class label')

        # Salvataggio plot
        today       = get_today_string()
        filename    = f'{today}_{model_name}_umap_{n_components}d_projection.png'
        filepath    = GlobalPaths.PLOT_MANIFOLD_LEARNING / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=600)
        plt.close(fig)
        log.info(f"[✓] Plot salvato in: {filepath}")

    def main(self):
        self.project_data()
        #self.save_projected_data()
        self.plot_projection()
        self.classify_data()
    
    def __del__(self):
        print('\nDestructor called for the class Model')

if __name__ == '__main__':
    umap_processor = myUMAP()
    umap_processor.main()