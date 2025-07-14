import  torch
import  os
import  matplotlib.pyplot   as      plt
from    datetime            import  datetime
from    pathlib             import  Path
from    dataclasses         import  dataclass, field
from    typing              import  List
from    sklearn.metrics     import  precision_score, recall_score, f1_score, roc_auc_score


class GlobalPaths:
    # exoplanets/
    HOME                = Path.home() / 'Desktop'                   # automatico: /home/<username>/Desktop
    MAIN                = HOME / 'exoplanets' / 'main'              # main/
    CODE                = MAIN / 'code'                             # main/code/
    DATA                = MAIN / 'data'                             # main/data/
    OUTPUT_FILES        = MAIN / 'output_files'                     # main/output_files/
    TRAINED_MODELS      = MAIN / 'trained_models'                   # main/trained_models/
    # output_files/
    TRAINING_METRICS    = OUTPUT_FILES / 'training_metrics'         # main/output_files/training_metrics/
    TRAINING_METRICS_FEATURE_EXTRACTOR = TRAINING_METRICS / 'feature_extractor'
    # data/
    MAIN_DATASETS           = DATA / 'main_datasets'
    CSV                     = MAIN_DATASETS / 'csv_format'
    DATA_NUMPY              = MAIN_DATASETS / 'numpy_format_split_80_20'
    FEATURES_STEP1_CNN      = DATA / 'features_step1_cnn'
    FEATURES_STEP2_TSNE     = DATA / 'features_step2_tsne'
    FEATURES_STEP2_MLP      = DATA / 'features_step2_mlp'
    FEATURES_STEP2_MANIFOLD = DATA / 'features_step2_manifold'
    PLATO_DATA_TABLES       = DATA / 'plato_data' / 'table_format'
    PLATO_DATA_NUMPY        = DATA / 'plato_data' / 'numpy_format'
    # code/
    DATASET             = CODE / 'dataset'                          # main/code/dataset
    FEATURE_EXTRACTION  = CODE / 'feature_extraction'               # main/code/feature_extraction
    UTILS               = CODE / 'utils'                            # main/code/utils/
    CONFIG              = CODE / 'config'
    # plot/
    PLOT_DEBUG              =   OUTPUT_FILES / 'plot_debug'
    PLOT_MANIFOLD_LEARNING  =   OUTPUT_FILES / 'plot_manifold_learning'
    PLOT_TSNE               =   OUTPUT_FILES / 'plot_tsne'
    # config files
    config_dataset_csv_file         =   'config_dataset.yaml'
    config_feature_extractor_file   =   'config_feature_extractor.yaml'
    config_vgg_file                 =   'config_vgg.yaml'
    config_resnet_file              =   'config_resnet.yaml'
    config_data_preparation         =   'config_data_preparation.yaml'
    config_manifold_learning        =   'config_manifold_learning.yaml'
    #TODO Put config filenames here
    
    # dataset filenames
    plato_parameters_ftr_file             =   'sim00465_06899_Parameters.ftr'
    plato_parameters_conts_ftr_file       =   'sim00465_06899_Parameters_Conts.ftr'
    plato_all_parameters_ftr_file         =   'sim00465_06899_AllParameters.ftr'
    plato_all_parameters_conts_ftr_file   =   'sim00465_06899_AllParameters_Conts.ftr'
    plato_fitted_events_ftr_file          =   'join_FittedEvents_AllParameters.ftr'
    plato_fitted_events_pkl_file          =   'Fitted_events_concatenate.pkl'

@dataclass
class TrainingMetrics:
    epochs:     List[int]   = field(default_factory=list)
    loss:       List[float] = field(default_factory=list)
    precision:  List[float] = field(default_factory=list)
    recall:     List[float] = field(default_factory=list)
    f1:         List[float] = field(default_factory=list)
    auc_roc:    List[float] = field(default_factory=list)

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
    
    def log_loss(self, epoch: int, loss: float):
        """
            Registra la loss all'epoca corrente
            Task: regressione
        """
        self.epochs.append(epoch)
        self.loss.append(loss)

    def print_last_classification(self):
        """
            Stampa i valori dell’ultima epoca
            Task: classificazione
        """
        print(f"Epoch {self.epochs[-1]} — Loss: {self.loss[-1]:.4f}, Precision: {self.precision[-1]:.3f}, Recall: {self.recall[-1]:.3f}, F1: {self.f1[-1]:.3f}, AUC: {self.auc_roc[-1]:.3f}")

    def print_last_regression(self):
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

    def compute_and_log_classification_metrics(self, y_true, y_pred, y_proba=None, epoch=0, loss=0.0, model_supports_proba=False):
        """
            Calcola le metriche di classificazione (precision, recall, f1, auc) e le salva.
        """
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Calcolo AUC se possibile
        auc = 0.0
        if model_supports_proba and y_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except:
                auc = 0.0

        self.log(epoch=epoch, loss=loss, precision=precision, recall=recall, f1=f1, auc=auc)
        self.print_last_classification()

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