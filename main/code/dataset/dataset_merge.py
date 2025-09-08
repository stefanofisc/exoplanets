import  sys
import  torch
from    pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from    utils               import  GlobalPaths
from    logger              import log

def merge_datasets(train_file: Path, test_file: Path, output_file: Path):
    """
        Unisce due dataset PyTorch in formato tensoriale e salva il risultato in un unico file .pt.

        Parameters
        ----------
        train_file : Path
            Path al file .pt del dataset di training (es. "kepler_q1-q17_dr25_multiclass_train_split.pt").
        test_file : Path
            Path al file .pt del dataset di test (es. "kepler_q1-q17_dr25_multiclass_test_split.pt").
        output_file : Path
            Path del file di output .pt in cui salvare il dataset unificato.

        Raises
        ------
        ValueError
            Se i formati dei due file non sono compatibili.
    """

    # --- Caricamento dataset ---
    train_data = torch.load(train_file)
    test_data  = torch.load(test_file)

    # --- Caso 1: dati come tuple (features, labels) ---
    if isinstance(train_data, tuple) and isinstance(test_data, tuple):
        train_X, train_y = train_data
        test_X, test_y   = test_data

        if train_X.shape[1:] != test_X.shape[1:]:
            raise ValueError("[!] I tensori delle feature hanno shape incompatibili.")
        if train_y.dim() != test_y.dim():
            raise ValueError("[!] I tensori delle label hanno dimensioni incompatibili.")

        merged_X    = torch.cat([train_X, test_X], dim=0)
        merged_y    = torch.cat([train_y, test_y], dim=0)
        merged_data = (merged_X, merged_y)

    # --- Caso 2: dati come dict {"features": ..., "labels": ...} ---
    elif isinstance(train_data, dict) and isinstance(test_data, dict):
        merged_data = {
            "features": torch.cat([train_data["features"], test_data["features"]], dim=0),
            "labels":   torch.cat([train_data["labels"], test_data["labels"]], dim=0),
        }

    else:
        raise ValueError("[!] Invalid file format .pt. Expected tuple o dict.")

    # --- Salvataggio output ---
    torch.save(merged_data, output_file)
    log.debug(f"[✓] Merged dataset saved to {output_file}")


if __name__ == '__main__':
    merge_datasets(
    Path(GlobalPaths.MAIN_DATASETS / "tensor_format_split_80_20" / "kepler_q1-q17_dr25_multiclass_train_split.pt"),
    Path(GlobalPaths.MAIN_DATASETS / "tensor_format_split_80_20" / "kepler_q1-q17_dr25_multiclass_test_split.pt"),
    Path(GlobalPaths.MAIN_DATASETS / "tensor_format" / "kepler_q1-q17_dr25_multiclass_train_test_split.pt")
    )