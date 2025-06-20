import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

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
    FEATURES_STEP2_MLP = DATA / 'features_step2_mlp'
    # code/
    DATASET = CODE / 'dataset'                      # main/code/dataset
    FEATURE_EXTRACTION = CODE / 'feature_extraction'# main/code/feature_extraction
    UTILS = CODE / 'utils'                          # main/code/utils/
    CONFIG = CODE / 'config'

@dataclass
class TrainingMetrics:
    epochs: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1: List[float] = field(default_factory=list)
    auc_roc: List[float] = field(default_factory=list)

    def log(self, epoch, loss, precision, recall, f1, auc):
        """
            Aggiunge i valori di una singola epoca
            Task: classificazione
        """
        self.epochs.append(epoch)
        self.loss.append(loss)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.auc_roc.append(auc)
    
    def log(self, epoch: int, loss: float):
        """
            Registra la loss all'epoca corrente
            Task: regressione
        """
        self.epochs.append(epoch)
        self.loss.append(loss)

    def print_last(self):
        """
            Stampa i valori dell’ultima epoca
            Task: classificazione
        """
        print(f"Epoch {self.epochs[-1]} — Loss: {self.loss[-1]:.4f}, Precision: {self.precision[-1]:.3f}, Recall: {self.recall[-1]:.3f}, F1: {self.f1[-1]:.3f}, AUC: {self.auc_roc[-1]:.3f}")

    def print_last(self):
        """
            Stampa i valori dell’ultima epoca
            Task: regressione
        """
        print(f"[Epoch {self.epochs[-1]}] Loss: {self.loss[-1]:.6f}")

    def plot_metrics(self, output_path: str, model_name: str, optimizer: str, num_epochs: int, df_name : str):
        """
          Salva i plot delle metriche con un nome file coerente con lo stile:
          YYYY-MM-DD_<model_name>_<optimizer>_<num_epochs>_<df_name>_<metric>.png

          In questo modo, rendo il formato del filename di output coerente con quello
          relativo alle caratteristiche estratte, salvate in features_step1_cnn.

          Task: classificazione
        """
        os.makedirs(output_path, exist_ok=True)  # crea la directory se non esiste
        today = get_today_string()
        metrics = ['loss', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(self.epochs, getattr(self, metric), label=metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'Training {metric.capitalize()} Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = f"{today}_{model_name}_{optimizer}_{num_epochs}_{df_name}_{metric}.png"
            plt.savefig(os.path.join(output_path, filename), dpi=300)
            plt.close()
    
    def plot_loss(self, output_path: str = None, filename: str = None):
        """
            Plotta la loss in funzione delle epoche
            Task: regressione
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs, self.loss, label='MSE Loss', color='tab:blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            path = os.path.join(output_path, filename)
            plt.savefig(path, dpi=300)
            print(f"[✓] Loss curve saved to: {path}")

        plt.close()


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