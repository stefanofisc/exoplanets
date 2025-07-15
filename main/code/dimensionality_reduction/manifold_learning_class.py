import  yaml
import  sys
import  pandas              as      pd
import  numpy               as      np
import  matplotlib.pyplot   as      plt
import  plotly.express      as      px
from    sklearn.manifold    import  Isomap
from    dataclasses         import  dataclass
from    pathlib             import  Path
from    typing              import  Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths, get_today_string
from    logger              import  log

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'dataset'))
from    dataset_manifold_learning   import DatasetManifoldLearning


@dataclass
class DatasetConfig:
    filename_samples:   str
    filename_labels:    str

@dataclass
class EmbeddingConfig:
    algorithm:          str
    n_components:       int
    #NOTE Maybe you'll change in future implementations as these are Isomap specific parameters
    n_neighbors:        int
    max_iter:           int
    radius:             Optional[float] = None
    eigen_solver:       Optional[str]   = 'auto'
    tol:                Optional[int]   = 0
    path_method:        Optional[str]   = 'auto'
    neighbors_algorithm:Optional[str]   = 'auto'
    metric:             Optional[str]   = 'minkowski'
    p:                  Optional[int]   = 2
    n_jobs:             Optional[int]   = 1

@dataclass
class OutputConfig:
    save_features:      bool
    save_plot:          bool

@dataclass
class InputVariablesManifoldLearning:
    _dataset:           DatasetConfig
    _embedding:         EmbeddingConfig
    _output:            OutputConfig

    @classmethod
    def get_input_hyperparameters(cls, filename: str):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        dataset_conf    = DatasetConfig(**config.get('dataset', {}))
        embedding_conf  = EmbeddingConfig(**config.get('embedding', {}))
        output_conf     = OutputConfig(**config.get('output', {}))

        return cls(
            _dataset    = dataset_conf,
            _embedding  = embedding_conf,
            _output     = output_conf
        )

class ManifoldLearning:
    def __init__(self):
        # Initialize the hyperparameters object 
        self.__manifold_learning_hyperparameters_object = self.__init_manifold_learning_hyperparameters()

        self.__extracted_features   = []    # output of the feature extraction module 
        self.__extracted_labels     = []    # initialized when plot_features = True
        self.__projected_features   = []    # output of the dim. red. module

        self.__embedding            = self.__init_embedding()
        self.__dataset              = self.__init_dataset()
    
    def __init_manifold_learning_hyperparameters(self):
        return InputVariablesManifoldLearning.get_input_hyperparameters(
            GlobalPaths.CONFIG / GlobalPaths.config_manifold_learning
            )

    def __init_embedding(self):
        """
            Initialize the sklearn.manifold object.
            Input parameters are defined in the config_manifold_learning.yaml file, within the 
            embedding object.
        """
        algorithm   = self.__manifold_learning_hyperparameters_object._embedding.algorithm
        methods     = {
            "isomap": Isomap(
                n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components,
                n_neighbors     = self.__manifold_learning_hyperparameters_object._embedding.n_neighbors,
                radius          = self.__manifold_learning_hyperparameters_object._embedding.radius,
                eigen_solver    = self.__manifold_learning_hyperparameters_object._embedding.eigen_solver,
                tol             = self.__manifold_learning_hyperparameters_object._embedding.tol,
                max_iter        = self.__manifold_learning_hyperparameters_object._embedding.max_iter,
                path_method     = self.__manifold_learning_hyperparameters_object._embedding.path_method,
                neighbors_algorithm = self.__manifold_learning_hyperparameters_object._embedding.neighbors_algorithm,
                metric          = self.__manifold_learning_hyperparameters_object._embedding.metric,
                p               = self.__manifold_learning_hyperparameters_object._embedding.p,
                n_jobs          = -1,
                )#,
            #"lle": LocalLinearEmbedding() 
        }
        return methods.get(algorithm)
    
    def __init_dataset(self):
        return DatasetManifoldLearning(
            self.__manifold_learning_hyperparameters_object._dataset
        )
    
    def __get_training_samples(self):
        self.__extracted_features, self.__extracted_labels, *_ = self.__dataset.get_training_test_samples()

    def __define_output_filename(self):
        filename_samples = self.__manifold_learning_hyperparameters_object._dataset.filename_samples
        n_components     = self.__manifold_learning_hyperparameters_object._embedding.n_components
        algorithm        = self.__manifold_learning_hyperparameters_object._embedding.algorithm
        prefix           = (filename_samples).split('.npy')[0]
        suffix           = f'{n_components}d_{algorithm}'

        output_filename_base = f'{prefix}_{suffix}'

        return (
            GlobalPaths.FEATURES_STEP2_MANIFOLD /
            f'{output_filename_base}.npy'
          )

    def project_data(self):
        #NOTE.  La logica del caricamento dei dati di training-test va riconsiderata. Qui stiamo caricando il dataset 
        #       definito in ${filename_samples}. Riflettere su cosa cambierà quando lo integreremo in dimensionality_reduction
        #NOTE.  Salva projected features in un file .npy come in altri moduli
        self.__get_training_samples()     
        self.__projected_features   = self.__embedding.fit_transform(self.__extracted_features)
    
    def plot_projected_data(self):
        #NOTE.  Possibilità di visualizzare gli event-id ad ogni punto proiettato. Grafico interattivo. 
        """
            Plot della proiezione dei dati nello spazio 2D o 3D, a seconda del valore di n_components.
        
            Input:
                - labels (np.ndarray): array di etichette di classe da usare per colorare i punti
        """
        today           = get_today_string()
        n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components
        algorithm       = self.__manifold_learning_hyperparameters_object._embedding.algorithm

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
                c=self.__extracted_labels,
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
                c=self.__extracted_labels,
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
        print(f"[✓] {n_components}D projection saved to: {filepath}")

    def plot_projected_data_interactive_html(self):
        """
            Save an interactive 3D plot (with zoom and rotation) in .html format. The plot will be
            opened in your browser.
        """
        today           = get_today_string()
        n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components
        algorithm       = self.__manifold_learning_hyperparameters_object._embedding.algorithm

        if n_components != 3:
            raise ValueError("[!] This method is intended for 3D projection only (n_components = 3)")

        projected = self.__projected_features
        df = pd.DataFrame({
            'Component 1':  projected[:, 0],
            'Component 2':  projected[:, 1],
            'Component 3':  projected[:, 2],
            'Label':        self.__extracted_labels
        })

        fig = px.scatter_3d(
            df,
            x='Component 1',
            y='Component 2',
            z='Component 3',
            color='Label',
            opacity=0.7,
            title=f'{algorithm.upper()} Projection (3D)'
        )

        filename = f'{today}_{algorithm}_3d_projection.html'
        filepath = GlobalPaths.OUTPUT_FILES / GlobalPaths.PLOT_MANIFOLD_LEARNING / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(filepath))  # salva file .html

        print(f"[✓] 3D interactive plot saved to: {filepath}")

    def save_projected_data(self):
        """
            Save the projected feature vectors in data/features_step2_manifold/.
            A warning is raised and feature vectors are not saved when output filename already exists
        """
        filepath_base = self.__define_output_filename()

        if filepath_base.exists():
            log.warning(f'[!] File {filepath_base.name} already exists. Projected features not saved to avoid overwriting.')
            return

        np.save(filepath_base.with_name(filepath_base.name), self.__projected_features)
        log.info(f'[✓] Projected features saved to {filepath_base}')      
        
    def main(self):
        """
            Main execution.
                1. Project data in the manifold
                2. Plot projected features
                3. Save projected feature vectors in features_step2_manifold
        """
        self.project_data()

        if self.__manifold_learning_hyperparameters_object._output.save_plot == True:
            n_components = self.__manifold_learning_hyperparameters_object._embedding.n_components
            if n_components == 2:
                self.plot_projected_data()
            elif n_components == 3:
                self.plot_projected_data_interactive_html()
            else:
                log.error(f'In class ManifoldLearning.main(). Cannot plot data in {n_components}d')
        
        if self.__manifold_learning_hyperparameters_object._output.save_features == True:
            self.save_projected_data()

    def __del__(self):
        log.info('Destructor for class ManifoldLearning')

if __name__ == '__main__':
    m = ManifoldLearning()
    m.main()
