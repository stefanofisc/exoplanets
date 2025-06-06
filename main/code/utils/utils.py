from datetime import datetime
from pathlib import Path

class GlobalPaths:
    HOME = Path.home() / 'Desktop'  # automatico: /home/<username>/Desktop
    PROJECT_ROOT = HOME / 'exoplanets' / 'main'
    CODE = PROJECT_ROOT / 'code'
    DATASET = CODE / 'dataset'
    UTILS = CODE / 'utils'

def get_today_string():
    """
        Resituisce la data odierna in formato YYYY-MM-DD.
    """
    return datetime.today().strftime('%Y-%m-%d')