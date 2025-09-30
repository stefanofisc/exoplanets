import  pandas  as      pd
from    dataset import  GlobalPaths, PathConfigDataset, Dataset, yaml, log

def main_dataset_class():
    """Main function for testing dataset functionality."""
    # Load configuration
    with open(GlobalPaths.CONFIG / "config_dataset.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Carica il dataset CSV
    df = pd.read_csv(PathConfigDataset.CSV / config["dataset_filename"])
    dataset_handler = Dataset(df)

    # Test class counting
    dataset_handler.count_classes()
    #dataset_handler.count_classes(dataframe='training')
    #dataset_handler.count_classes(dataframe='test')
    
    # Test tensor access
    x_train, y_train, x_test, y_test = dataset_handler.get_training_test_samples()
    log.debug('\n Testing the method for loading the training test samples')
    log.debug(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    log.debug(f"X_test:  {x_test.shape}, y_test:  {y_test.shape}")
    
    del dataset_handler, df
    
if __name__ == "__main__":
    main_dataset_class()