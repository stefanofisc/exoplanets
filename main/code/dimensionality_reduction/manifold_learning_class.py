import  yaml
import  sys
import  matplotlib.pyplot   as      plt
from    mpl_toolkits.mplot3d import Axes3D  # Necessario per il plot 3D
from    sklearn.manifold    import  Isomap
from    dataclasses         import  dataclass
from    pathlib             import  Path
from    typing              import  List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths, get_today_string
from    logger              import  Logger

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from    dataset_manifold_learning   import DatasetManifoldLearning


@dataclass
class DatasetConfig:
    filename_samples: str
    filename_labels: Optional[str] = None

@dataclass
class EmbeddingConfig:
    algorithm: str
    n_components: int
    #variable: Optional[type] = value

@dataclass
class InputVariablesManifoldLearning:
    _dataset: DatasetConfig
    _embedding: EmbeddingConfig

    @classmethod
    def get_input_hyperparameters(cls, filename: str):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        dataset_conf    = DatasetConfig(**config.get('dataset', {}))
        embedding_conf  = EmbeddingConfig(**config.get('embedding', {}))

        return cls(
            _dataset    = dataset_conf,
            _embedding  = embedding_conf
        )

class ManifoldLearning:
    def __init__(self):
        self.__log = Logger()

        # Initialize the hyperparameters object 
        self.__manifold_learning_hyperparameters_object = self.__init_manifold_learning_hyperparameters()
        
        self.__embedding    = self.__init_embedding()
        self.__dataset      = self.__init_dataset()
        
        self.__projected_features = []

    
    def __init_manifold_learning_hyperparameters(self):
        return InputVariablesManifoldLearning.get_input_hyperparameters(
            GlobalPaths.CONFIG / GlobalPaths.config_manifold_learning
            )

    def __init_embedding(self):
        algorithm       = self.__manifold_learning_hyperparameters_object._embedding.algorithm
        n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components
        
        methods = {
            "isomap": Isomap(n_components = n_components, n_jobs=-1)#,
            #"lle": LocalLinearEmbedding() 
        }
        return methods.get(algorithm)
    
    def __init_dataset(self):
        return DatasetManifoldLearning(
            self.__manifold_learning_hyperparameters_object._dataset
        )
    
    def project_data(self):
        #NOTE.  La logica del caricamento dei dati di training-test va riconsiderata. Qui stiamo caricando il dataset 
        #       definito in ${filename_samples}. Riflettere su cosa cambierà quando lo integreremo in dimensionality_reduction
        #NOTE.  Salva projected features in un file .npy come in altri moduli
        X, *_                       = self.__dataset.get_training_test_samples()     
        self.__projected_features   = self.__embedding.fit_transform(X)
    
    def plot_projected_data(self):
        """
        Plot della proiezione dei dati nello spazio 2D o 3D, a seconda del valore di n_components.
        
        Args:
            labels (np.ndarray): array di etichette di classe da usare per colorare i punti
        """
        today           = get_today_string()
        n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components
        algorithm       = self.__manifold_learning_hyperparameters_object._embedding.algorithm
        _, labels, *_   = self.__dataset.get_training_test_samples()

        # Controllo sulla dimensionalità proiettata
        if self.__projected_features.shape[1] != n_components:
            raise ValueError(f"[!] Dimensione dei dati proiettati ({self.__projected_features.shape[1]}) diversa da n_components ({n_components})")

        # Creazione figura e asse
        fig = plt.figure(figsize=(10, 7))

        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                self.__projected_features[:, 0],
                self.__projected_features[:, 1],
                c=labels,
                cmap='viridis',
                edgecolors='k',
                alpha=0.7
            )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(f'{algorithm.upper()} Projection (2D)')

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
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(f'{algorithm.upper()} Projection (3D)')

        else:
            raise ValueError(f"[!] Visualizzazione supportata solo per n_components = 2 o 3, ma è {n_components}")

        # Colore classi
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Class label')

        # Salvataggio plot
        filename = f'{today}_{algorithm}_{n_components}d_projection.png'
        filepath = GlobalPaths.OUTPUT_FILES / GlobalPaths.PLOT_MANIFOLD_LEARNING / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=1200)
        plt.close()
        print(f"[✓] Proiezione {n_components}D salvata in: {filepath}")

    def save_projected_data(self):
        pass

    def __del__(self):
        self.__log.info('Destructor for class ManifoldLearning')

if __name__ == '__main__':
    m = ManifoldLearning()
    m.project_data()
    m.plot_projected_data()
