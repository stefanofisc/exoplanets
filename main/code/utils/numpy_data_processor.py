import  numpy               as np
import  matplotlib.pyplot   as plt
from    typing              import Tuple, Optional, Dict

class NumpyDataProcessor:
    """
    Classe per la gestione e la visualizzazione di array numpy contenenti global views.
    Include normalizzazione, istogrammi e plotting di esempi.

    Esempio d'uso:
    --------------
        processor = NumpyDataProcessor(X_train_np)
        X_norm, info = processor.normalize_global_views_to_median1_min0(per_sample=True)
        processor.plot_histograms_before_after(X_norm)
        processor.plot_sample_global_views(X_norm)
    """

    def __init__(self, data: np.ndarray):
        """
        Parametri
        ----------
        data : np.ndarray
            Array numpy di shape (N, L), dove N = numero di campioni, L = lunghezza vettori.
        """
        self.data = np.array(data)
        if self.data.ndim != 2:
            raise ValueError(f"Expected 2D array (N, L), got {self.data.ndim}D.")
        self.N, self.L = self.data.shape
        print(f"[INFO] Dataset initialized with shape: {self.data.shape}")

    # -----------------------------------------------------
    # 1 Normalization
    # -----------------------------------------------------
    def normalize_global_views_to_median1_min0(
        self,
        per_sample: bool = True,
        clip: bool = True,
        eps: float = 1e-8,
        return_transform: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Normalizza i vettori in modo che la mediana -> 1 e la profondità minima -> 0.
        Può essere eseguita per campione o su tutto il dataset.
        """
        X = self.data.copy()

        if per_sample:
            medians = np.median(X, axis=1)          # (N,)
            mins = np.min(X, axis=1)                # (N,)
            scales = medians - mins                 # (N,)
            scales = np.where(np.abs(scales) < eps, eps, scales)

            medians_b = medians[:, None]
            scales_b = scales[:, None]

            X_norm = (X - medians_b) / scales_b + 1.0

            transform_info = {
                "per_sample": True,
                "medians": medians,
                "mins": mins,
                "scales": scales
            }

        else:
            med_global = np.median(X)
            min_global = np.min(X)
            scale_global = med_global - min_global
            if abs(scale_global) < eps:
                scale_global = eps
            X_norm = (X - med_global) / scale_global + 1.0

            transform_info = {
                "per_sample": False,
                "median_global": med_global,
                "min_global": min_global,
                "scale_global": scale_global
            }

        if clip:
            X_norm = np.clip(X_norm, 0.0, 1.0)

        print(f"[INFO] Normalization complete. Range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")

        if return_transform:
            return X_norm, transform_info
        else:
            return X_norm, None

    # -----------------------------------------------------
    # 2 Histograms before and after scaling
    # -----------------------------------------------------
    def plot_histograms_before_after(self, X_norm: np.ndarray, filename: str, bins: int = 50, dpi: int = 600):
        """
            Mostra istogrammi di distribuzione dei valori prima e dopo la normalizzazione.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        axes[0].hist(self.data.flatten(), bins=bins, color='gray', alpha=0.7)
        axes[0].set_title("Before Normalization")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(X_norm.flatten(), bins=bins, color='green', alpha=0.7)
        axes[1].set_title("After Normalization")
        axes[1].set_xlabel("Value")

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)

    # -----------------------------------------------------
    # 3 Plotting normalized global views
    # -----------------------------------------------------
    def plot_sample_global_views(self, X_norm: np.ndarray, X_original: np.ndarray, filename: str, num_samples: int = 6, dpi: int = 600):
        """
            Show 6 normalized global views (2x3 subplot).
        """
        if X_norm.ndim != 2:
            raise ValueError("X_norm deve avere shape (N, L)")
        if X_norm.shape[0] < num_samples:
            raise ValueError("Non ci sono abbastanza campioni per il plot richiesto.")

        indices = np.random.choice(X_norm.shape[0], size=num_samples, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            axes[i].plot(X_norm[idx], color="#21918c", label='Normalized')
            axes[i].plot(X_original[idx], label='Original')
            #axes[i].set_ylim(0, 1)
            axes[i].set_title(f"Sample {idx}", fontsize=12)
            axes[i].set_xlabel("Time bins")
            axes[i].set_ylabel("Normalized flux")
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)
