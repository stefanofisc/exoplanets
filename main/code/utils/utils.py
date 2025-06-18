from datetime import datetime
from pathlib import Path
import torch

class GlobalPaths:
    # exoplanets/
    HOME = Path.home() / 'Desktop'                  # automatico: /home/<username>/Desktop
    MAIN = HOME / 'exoplanets' / 'main'             # main/
    CODE = MAIN / 'code'                            # main/code/
    DATA = MAIN / 'data'                            # main/data/
    OUTPUT_FILES = MAIN / 'output_files'            # main/output_files/
    TRAINED_MODELS = MAIN / 'trained_models'        # main/trained_models/
    # data/
    CSV = DATA / 'main_datasets' / 'csv_format'
    FEATURES_STEP1_CNN = DATA / 'features_step1_cnn'
    FEATURES_STEP2_TSNE = DATA / 'features_step2_tsne'
    # code/
    DATASET = CODE / 'dataset'                      # main/code/dataset
    FEATURE_EXTRACTION = CODE / 'feature_extraction'# main/code/feature_extraction
    UTILS = CODE / 'utils'                          # main/code/utils/
    CONFIG = CODE / 'config'
    
def get_today_string():
    """
        Resituisce la data odierna in formato YYYY-MM-DD.
    """
    return datetime.today().strftime('%Y-%m-%d')

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device