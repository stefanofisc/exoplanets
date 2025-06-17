import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils import GlobalPaths

@dataclass
class InputVariablesTSNE:
    # define type of input data
    _ncomp: int
    _perplexity: int
    _max_iter: int
    _extracted_features_filename: str

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _ncomp=config.get('ncomp', 2),                             
            _perplexity=config.get('perplexity', 25),
            _max_iter=config.get('max_iter', 2000),
            _extracted_features_filename=config.get('extracted_features_filename', None),
            )
    
    # define get and set methods
    
class myTSNE:
    def __init__(self):
        self.__tsne_hyperparameters_object = None
        self.__extracted_features = []      # output of the feature extraction module 
        self.__projected_features = []      # output of the dim. red. module
    
        self.__init_tsne_hyperparameters()
        self.__load_extracted_features()

    
    def __init_tsne_hyperparameters(self):
        self.__tsne_hyperparameters_object = InputVariablesTSNE.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_tsne.yaml')

    def __load_extracted_features(self):
        self.__extracted_features = np.load(GlobalPaths.FEATURES_STEP1_CNN / self.__tsne_hyperparameters_object._extracted_features_filename)

    def __project_features(self):
        print("[INFO] Projecting features with t-SNE. This may take a while...")
        # Define a sklearn.TSNE object
        tsne = TSNE(
            n_components=self.__tsne_hyperparameters_object._ncomp, 
            random_state=42, 
            perplexity=self.__tsne_hyperparameters_object._perplexity, 
            max_iter=self.__tsne_hyperparameters_object._max_iter
            )
        
        self.__projected_features = tsne.fit_transform(self.__extracted_features)
        print("[✓] Projection completed.")
    

    def __save_projected_feature_vectors(self):
        prefix = (self.__tsne_hyperparameters_object._extracted_features_filename).split('.npy')[0]
        suffix = f'{self.__tsne_hyperparameters_object._ncomp}d_tsne.npy'

        filepath_base = (
            GlobalPaths.FEATURES_STEP2_TSNE /
            f'{prefix}_{suffix}'
          )
        
        np.save(filepath_base.with_name(filepath_base.name), self.__projected_features)

        print(f'[✓] Projected features saved to {filepath_base}')      

    def main(self):
        self.__project_features()
        self.__save_projected_feature_vectors()
        print('myTSNE class. Ending...')


    def __del__(self):
        print('\nDestructor called for the class myTSNE')

if __name__ == '__main__':
    tsne = myTSNE()
    tsne.main()
