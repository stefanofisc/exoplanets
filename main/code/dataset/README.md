# Dataset Processing Module

This module provides utilities for managing datasets used in machine learning pipelines. It supports reading, preprocessing, splitting, encoding, and saving data in CSV and PyTorch tensor formats.

## Structure

* `dataset.py` – Main configuration and shared dataset logic.
* `dataset_backup.py` – Legacy or backup logic for dataset operations.
* `dataset_classifier.py` – Dataset handler tailored for classification tasks.
* `dataset_mlp.py` – Dataset handler customized for MLP-based models.
* `tensor_data_handler.py` – Utilities for saving/loading tensor datasets.

## Features

* Label encoding with optional predefined mappings
* Train/test split with configurable test size
* Class distribution printing
* Saving data to CSV and `.pt` tensor files
* Loading tensor datasets for PyTorch models

## Configuration File (YAML)

```yaml
initialize_from_scratch: true
dataset_splitting: true
test_size: 0.2
dataset_save_split_csv: true
train_df_filename: "train.csv"
test_df_filename: "test.csv"
dataset_save_split_tensors: true
train_tensor_path: "train.pt"
test_tensor_path: "test.pt"
```

## Requirements

* Python 3.8+
* PyTorch
* pandas
* scikit-learn
* PyYAML
* numpy
