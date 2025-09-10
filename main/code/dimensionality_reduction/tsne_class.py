import  sys
import  yaml
import  numpy               as      np
import  matplotlib.pyplot   as      plt
from    pathlib             import  Path
from    sklearn.manifold    import  TSNE
from    dataclasses         import  dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths
from    logger              import  log

@dataclass
class InputVariablesTSNE:
    # define type of input data
    _ncomp:                         int
    _perplexity:                    int
    _max_iter:                      int
    _extracted_features_filename:   str
    _plot_features:                 bool

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _ncomp                      =   config.get('n_comp', 2),                             
            _perplexity                 =   config.get('perplexity', 25),
            _max_iter                   =   config.get('max_iter', 2000),
            _extracted_features_filename=   config.get('extracted_features_filename', None),
            _plot_features              =   config.get('plot_features', False)
            )
        
class myTSNE:
    def __init__(self):
        self.__tsne_hyperparameters_object  = None
        self.__extracted_features           = []    # output of the feature extraction module 
        self.__extracted_labels             = []    # initialized when plot_features = True
        self.__projected_features           = []    # output of the dim. red. module
        self.__output_filename_base         = ''    # base filename structure, without format extension. Used for projected_features and fig_filename
    
        self.__init_tsne_hyperparameters()
        self.__load_extracted_features()

    def __init_tsne_hyperparameters(self):
        self.__tsne_hyperparameters_object  = InputVariablesTSNE.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_tsne.yaml')

    def __load_extracted_features(self):
        self.__extracted_features           = np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__tsne_hyperparameters_object._extracted_features_filename)

        # Load labels when plot_features in config_tsne.yaml is True
        if self.__tsne_hyperparameters_object._plot_features:
            # Retrieve *_labels.npy filename from _extracted_features_filename, by replacing 'features' with 'labels'
            prefix                  = (self.__tsne_hyperparameters_object._extracted_features_filename).split('features')[0]
            suffix                  = 'labels.npy'
            labels_filename         = f'{prefix}{suffix}'
            self.__extracted_labels = np.load(GlobalPaths.FEATURES_STEP1_CNN / labels_filename)

    def __project_features(self):
        log.info("Projecting features with t-SNE. This may take a while...")
        # Define a sklearn.TSNE object
        tsne = TSNE(
            n_components    =   self.__tsne_hyperparameters_object._ncomp, 
            random_state    =   42, 
            perplexity      =   self.__tsne_hyperparameters_object._perplexity, 
            max_iter        =   self.__tsne_hyperparameters_object._max_iter
            )
        
        self.__projected_features = tsne.fit_transform(self.__extracted_features)
        log.info("[✓] Projection completed.")
    
    def __save_projected_feature_vectors(self):
        prefix  = (self.__tsne_hyperparameters_object._extracted_features_filename).split('.npy')[0]
        suffix  = f'{self.__tsne_hyperparameters_object._ncomp}d_tsne'
        
        self.__output_filename_base = f'{prefix}_{suffix}'

        filepath_base = (
            GlobalPaths.FEATURES_STEP2_TSNE /
            f'{self.__output_filename_base}.npy'
          )
        
        np.save(filepath_base.with_name(filepath_base.name), self.__projected_features)

        print(f'[✓] Projected features saved to {filepath_base}')      

    def __plot_tsne_representation(self, alpha=0.4, cmap='viridis'):
        fontsize    = 20
        resolution  = 1200
        n_components= self.__tsne_hyperparameters_object._ncomp

        xlabel='t-SNE Dimension 1'
        ylabel='t-SNE Dimension 2'
        # The input vector labels contains the true class labels and it is used to color the projected points
        fig = plt.figure(figsize=(10, 8))
        if n_components == 2:
            scatter = plt.scatter(
                self.__projected_features[:, 0], 
                self.__projected_features[:, 1], 
                c=self.__extracted_labels, 
                cmap=cmap, 
                alpha=alpha
                )
            plt.colorbar(scatter, label='Class Labels')
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

        elif n_components == 3:
            zlabel = 't-SNE Dimension 3'
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                self.__projected_features[:, 0],
                self.__projected_features[:, 1],
                self.__projected_features[:, 2],
                c=self.__extracted_labels,
                cmap=cmap,
                edgecolors='k',
                alpha=alpha
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

        else:
            raise ValueError(f"[!] Visualizzazione supportata solo per n_components = 2 o 3, ma è {n_components}")

        filepath_base = (
            GlobalPaths.PLOT_TSNE / 
            f'{self.__output_filename_base}.png'
          )

        plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
        plt.close()

    def main(self):
        self.__project_features()
        self.__save_projected_feature_vectors()
        if self.__tsne_hyperparameters_object._plot_features:
            self.__plot_tsne_representation()
        log.info('TSNE, ending...')

    def __del__(self):
        log.info('\nDestructor called for the class myTSNE')

"""
2025-09-10. 
    Questi tre metodi ti fanno plottare le proiezioni del t-SNE normalizzate, facendoti proprio
    vedere l'input su cui è addestrato il MLP. In future implementazioni, integra questo codice in questo
    modulo.
def __tmp_load_data(filename_features, filename_labels):
    x_train_numpy   = np.load(filename_features)
    labels          = np.load(filename_labels)
    return x_train_numpy, labels

def __tmp_normalize_data(x_train_numpy):
        #Normalize data to zero mean and unit variance
    epsilon = 1e-8  # offset to improve numerical stability. This prevents division by zero for features with zero std
    
    x_train_numpy_norm = (x_train_numpy - x_train_numpy.mean()) / (x_train_numpy.std() + epsilon )
    return x_train_numpy_norm

def __tmp_plot_normalized_data(x_train_numpy_norm, labels):
    fontsize    = 24
    resolution  = 1200

    xlabel='t-SNE Dimension 1'
    ylabel='t-SNE Dimension 2'
    # The input vector labels contains the true class labels and it is used to color the projected points
    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_train_numpy_norm[:, 0], 
        x_train_numpy_norm[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.6
        )

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    output_filename= '2025-08-27_resnet18_adam_80_kepler_dr25_train_features_2d_tsne_normalized_data'
    filepath_base = (
        GlobalPaths.PLOT_TSNE / 
        f'{output_filename}.png'
        )
    plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
    plt.close()
"""

if __name__ == '__main__':
    tsne = myTSNE()
    tsne.main()
