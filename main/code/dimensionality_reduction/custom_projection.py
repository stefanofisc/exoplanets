"""
    custom_projection.py

    Funzioni di visualizzazione per vettori 2D proiettati in uno spazio 3D,
    dove la terza coordinata (z) è fissata in base alla classe associata
    a ciascun campione.

    Autore: Stefano Fiscale
    Data: 2025-08-22
"""
import  sys
import  yaml
import  numpy               as     np
import  matplotlib.pyplot   as     plt
import  plotly.express      as     px
from    pathlib             import Path
from    dataclasses         import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths
from    logger              import log


@dataclass
class InputVariables:
    # define type of input data
    _features_filename: str
    #_labels_filename:   str
    _resolution:        int = 1200

    @classmethod
    def get_input_hyperparameters(cls, filename):
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _features_filename  =   config.get('features_filename', None),
            #_labels_filename    =   config.get('labels_filename', None),
            _resolution         =   config.get('output_plot_resolution', 1200)
            )

def __init_projection_hyperparameters():
    return InputVariables.get_input_hyperparameters(GlobalPaths.CONFIG / 'config_custom_projection.yaml')

def __init_output_plot_filename(features_filename: str):
    features_filename = features_filename.split('/')[1]
    prefix = features_filename.split('2d')[0]
    suffix = features_filename.split('2d')[1][:-4]  #slicing, remove .npy
    return f'{prefix}3d{suffix}' 

def __load_data(features_filename: str):
    x_filepath = (
        GlobalPaths.DATA / 
        f'{features_filename}'
    )
    features_filename = features_filename.split('/')[1]
    y_filepath = (
        GlobalPaths.FEATURES_STEP1_CNN /
        f"{features_filename.split('features')[0]}labels.npy"
    )
    #print(f'Features: {x_filepath}')
    #print(f'Labels: {y_filepath}')

    # --- Loading data ---
    X = np.load(x_filepath)
    Y = np.load(y_filepath)
    log.debug(f'Loaded vectors (X,Y) of shape {X.shape} and {Y.shape}')

    return X, Y

def plot_vectors_with_class_z(
    X,
    Y,
    z_values: dict = {0: 0, 1: 1, 2: 2},
    output_plot_filename: str = None,
    resolution: int = 1200,
    cmap: str = "viridis",
    alpha: float = 0.6,
    fontsize_labels: int = 14
):
    """
        Plotta vettori 2D (X) in uno spazio 3D, aggiungendo una coordinata z
        determinata dalla classe associata a ciascun campione.

        Parameters
        ----------
        X : np.ndarray
            Array di forma (N, 2), dove ogni riga è una coppia (x1, x2).
        Y : np.ndarray
            Array di forma (N,), con valori appartenenti all'insieme {0, 1, 2}.
        z_values : dict, optional
            Dizionario che associa ad ogni classe un valore z.
            Esempio: {0: 0, 1: 1, 2: 2}. Default: {0: 0, 1: 1, 2: 2}.
        cmap : str, optional
            Colormap per colorare i punti in base alla classe. Default: "viridis".
        alpha : float, optional
            Trasparenza dei punti. Default: 0.6.
        output_plot_filename : Path, optional
            il plot viene salvato in questo percorso
            (es. Path("plot.png")).

        Raises
        ------
        ValueError
            Se le dimensioni di X e Y non sono coerenti o X non ha forma (N, 2).
    """

    if X.shape[1] != 2:
        raise ValueError(f"[!] X deve avere shape (N, 2), invece ha {X.shape}.")
    if len(Y) != X.shape[0]:
        raise ValueError(f"[!] Y deve avere {X.shape[0]} elementi, invece ha {len(Y)}.")

    # --- Definizione coordinate z in base alla classe ---
    if z_values is None:
        z_values = {0: 0, 1: 1, 2: 2}

    Z = np.array([z_values[int(c)] for c in Y])

    # --- Costruzione matrice 3D ---
    X3D = np.c_[X, Z]

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X3D[:, 0],
        X3D[:, 1],
        X3D[:, 2],
        c=Y,
        cmap=cmap,
        alpha=alpha,
        edgecolors="k"
    )

    # Etichette assi
    ax.set_xlabel("x1", fontsize=fontsize_labels)
    ax.set_ylabel("x2", fontsize=fontsize_labels)
    ax.set_zlabel("Class-defined z", fontsize=fontsize_labels)

    # Colormap
    plt.colorbar(scatter, label="Class Labels")

    # --- Output ---
    filepath_base = (
        GlobalPaths.PLOT_CUSTOM_PROJECTION / 
        f'{output_plot_filename}.png'
    )

    plt.savefig(filepath_base.with_name(filepath_base.name), dpi=resolution)
    plt.close()

def plot_vectors_with_class_z_interactive(
    X,
    Y,
    z_values: dict = {0: 0, 1: 1, 2: 2},
    output_plot_filename: str = None,
    alpha: float = 0.6,
):
    """
    Plotta vettori 2D (X) in uno spazio 3D interattivo, aggiungendo una coordinata z
    determinata dalla classe associata a ciascun campione, e salva in formato .html.

    Parameters
    ----------
    X : np.ndarray
        Array di forma (N, 2), dove ogni riga è una coppia (x1, x2).
    Y : np.ndarray
        Array di forma (N,), con valori appartenenti all'insieme {0, 1, 2}.
    z_values : dict, optional
        Dizionario che associa ad ogni classe un valore z.
        Default: {0: 0, 1: 1, 2: 2}.
    output_plot_filename : str, optional
        Nome del file di output (es. "plot.html"). Se None, usa "plot_vectors.html".
    alpha : float, optional
        Trasparenza dei punti. Default: 0.6.

    Raises
    ------
    ValueError
        Se le dimensioni di X e Y non sono coerenti o X non ha forma (N, 2).
    """

    if X.shape[1] != 2:
        raise ValueError(f"[!] X deve avere shape (N, 2), invece ha {X.shape}.")
    if len(Y) != X.shape[0]:
        raise ValueError(f"[!] Y deve avere {X.shape[0]} elementi, invece ha {len(Y)}.")

    # --- Definizione coordinate z in base alla classe ---
    if z_values is None:
        z_values = {0: 0, 1: 1, 2: 2}
    Z = np.array([z_values[int(c)] for c in Y])

    # --- Mapping colori per classi ---
    color_map = {
        "0": "#3d014b",   # viola scuro
        "1": "#21918c",   # verde acqua
        "2": "#fde724"    # giallo brillante
    }

    # --- Plot con Plotly Express ---
    fig = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=Z,
        color=Y.astype(str),  # classi come stringhe per legenda leggibile
        opacity=alpha,
        color_discrete_map=color_map,
        labels={"x": "x1", "y": "x2", "z": "Class-defined z", "color": "Class Labels"}
    )

    # Titolo e layout
    fig.update_layout(
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="Class-defined z"
        )
    )

    # --- Output ---
    if output_plot_filename is None:
        output_plot_filename = "plot_vectors.html"

    filepath_base = (
        GlobalPaths.PLOT_CUSTOM_PROJECTION / 
        f'{output_plot_filename}.html'
    )
    fig.write_html(str(filepath_base))

    print(f"[✓] Plot interattivo salvato in {filepath_base}")

if __name__ == '__main__':
    input_variables = __init_projection_hyperparameters()
    X, Y            = __load_data(input_variables._features_filename)

    plot_vectors_with_class_z_interactive(
        X,
        Y,
        #resolution = input_variables._resolution,
        output_plot_filename=__init_output_plot_filename(input_variables._features_filename)
    )