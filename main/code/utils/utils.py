from datetime import datetime
from pathlib import Path

class GlobalPaths:
    # exoplanets/
    HOME = Path.home() / 'Desktop'                  # automatico: /home/<username>/Desktop
    MAIN = HOME / 'exoplanets' / 'main'             # main/
    CODE = MAIN / 'code'                            # main/code/
    DATA = MAIN / 'data'                            # main/data/
    OUTPUT_FILES = MAIN / 'output_files'            # main/output_files/
    # data/
    CSV = DATA / 'main_datasets' / 'csv_format'
    FEATURES_STEP1_CNN = DATA / 'features_step1_cnn'
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