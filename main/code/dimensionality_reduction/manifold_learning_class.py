import  yaml
import  sys
from    sklearn.manifold    import  Isomap
from    dataclasses         import  dataclass
from    pathlib             import  Path
from    typing              import  List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths
from    logger              import  Logger


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
        #self.__log.info(type(self.__dataset))   #DEBUG
        
        # Initialize the hyperparameters object 
        self.__manifold_learning_hyperparameters_object = self.__init_manifold_learning_hyperparameters()
        
        self.__embedding    = self.__init_embedding()
        #TODO self.__dataset      = self.__init_dataset()
    
    def __init_manifold_learning_hyperparameters(self):
        return InputVariablesManifoldLearning.get_input_hyperparameters(
            GlobalPaths.CONFIG / GlobalPaths.config_manifold_learning
            )

    def __init_embedding(self):
        algorithm       = self.__manifold_learning_hyperparameters_object._embedding.algorithm
        n_components    = self.__manifold_learning_hyperparameters_object._embedding.n_components
        
        if algorithm == 'isomap':
            self.__log.info('here we go Isomap')
            return Isomap(n_components = n_components)
        else:
            pass
            return None
    
    def __init_dataset(self):
        pass

    def __del__(self):
        self.__log.info('Destructor for class ManifoldLearning')

if __name__ == '__main__':
    m = ManifoldLearning()
