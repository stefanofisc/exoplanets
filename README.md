# Exoplanet Detection 🪐

## Introduction
A Deep Learning pipeline combining Convolutional Neural Networks, Dimensionality Reduction and Classifiers to perform a multi-class classification of NASA's Kepler and TESS Threshold-Crossing Events.

**NOTE**. Currently, any use of the code for publication purposes is not allowed without my consent (Stefano Fiscale).

---

## Dataset

**Download Link:** https://drive.google.com/file/d/18rUMHXPRWOwFYCNCZ-ph27aB-qbEPUtO/view?usp=sharing

## Pipeline Modules

1. **Dataset Preprocessing** - Creating or loading training-test split
2. **Feature Extraction** - Feature extraction from input signals with VGG-19, Resnet-18 or Resnet-34
3. **Dimensionality Reduction** - Projecting high-dimensional feature vectors into lower dimensional spaces
4. **Classification** - Classifying the projected feature vectors with linear/non-linear classifiers
## Execution Modes

To ensure code reusability and adaptability for other projects, the system is designed with independent, modular components that can be combined to create a complete ML pipeline.

Two execution modes are available:
* Individual Modules: Execute specific components independently for targeted analysis or debugging.
* Complete Pipeline: Automated execution of all modules in sequence for end-to-end processing.

## Configuration Files

Configuration files for individual modules are located in the `config/` directory. These files enable execution with custom parameters tailored to specific requirements.

For detailed parameter descriptions and usage instructions, refer to the respective `readme_<module_name>.md` files in each module directory.

**Note:** Parameters are loaded by individual modules and are also applied during Complete Pipeline execution.

## Usage
Below is an example of how each of the modules developed can be used.

### 1. Feature Extraction with Convolutional Neural Network (CNN)
This module allows you to train and test a Convolutional Neural Network (CNN) in order to extract the feature vectors from the last feature extraction layer of the model. The CNNs available are: VGG-19, Resnet-18, Resnet-34.
It is also possible to use the trained CNN as a classifier by setting it to inference mode.

Configuration files to define input parameters:
- config_dataset.yaml
- config_feature_extractor.yaml
- config_resnet.yaml (or config_vgg.yaml), depending on the CNN architecture you wish to use as the feature extractor)

```bash
conda activate <your_env_name>
cd main
python3 code/feature_extraction/feature_extractor.py > output_files/<name_of_output_file>.out
```

After execution, the following files will be generated:

- [In data/features_step1_cnn/]: Two .npy files containing the extracted feature vectors and the corresponding labels of the processed signals.
- [In output_files/]: A .out file listing all the operations performed during the execution.
- [In output_files/training_metrics/feature_extractor/]: Five .png files showing the evolution of evaluation metrics (Loss, AUC, Precision, Recall, F1-score) on the training and validation sets (if used).


### 2. Dimensionality Reduction
This module takes the feature vectors produced by the previous module and projects them into a subspace. It is possible to project data into low-dimensional spaces using the following methods:
1. t-Stochastic Neighbor Embedding;
2. Multi-Layer Perceptron addestrato per apprendere il mapping di t-SNE;
3. Parametric UMAP

2.1. Projecting feature vectors with t-SNE algorithm:
Configuration files to define input parameters:
- confid_tsne.yaml

```bash
cd main
python3 code/dimensionality_reduction/tsne_class.py > output_files/<name_of_output_file>.out
```

After execution, the following files will be generated:

- [In data/features_step2_tsne/]: A .npy file containing the feature vectors projected in the lower dimensional embedding by t-SNE;
- [In output_files/]: A .out file listing all the operations performed during the execution;
- [In output_files/plot_tsne/]: A .png file representing the projected feature vectors. Here's an example of three-dimensional projection of feature vectors produced by Resnet-18 on the Kepler Q1-Q17 Data Release 25 dataset: https://drive.google.com/file/d/1VufXdxVzRNSRHCqdGfdoSESjP7-KC0hi/view?usp=sharing

2.2. Projecting feature vectors with MLP:
Configuration files to define input parameters:
- confid_mlp.yaml

```bash
cd main
python3 code/dimensionality_reduction/mlp_class.py > output_files/<name_of_output_file>.out
```

2.3. Projecting feature vectors with Parametric UMAP:
Configuration files to define input parameters:
- config_dataset.yaml
- config_resnet.yaml (or config_vgg.yaml), depending on the CNN you want to use as encoder.
- confid_umap.yaml

```bash
cd main
python3 code/dimensionality_reduction/umap_class.py > output_files/<name_of_output_file>.out
```

### 3. Classification
<description of the module>
Configuration files to define input parameters:
- config_classifier.yaml
-

### Complete Pipeline Execution
**Note:** This function is still not available. We plan to build this module in future implementations.
```bash
python3 main_pipeline.py
```

### Individual Module Execution
Navigate to the specific module directory and refer to the corresponding README file for execution instructions. 

## Project Structure

```
│───main/
│   └───trained_models/                     # Output models trained
│   └───data/                               # Dataset storage
│   └───output_files/                       # output folder
│   └───code/                               # Individual processing modules
│       ├───classification/                 # Classification module
│       ├───config/                         # Configuration files
│       ├───dataset/                        # Data preprocessing module
│       ├───dimensionality_reduction/       # Dimensionality reduction module
│       ├───feature_extraction/             # Feature extraction module
│       │   ├───resnet/                     # ResNet18, ResNet34 class  
│       │   └───vgg/                        # VGG-19 class
│       └───utils/                          # library
```
## Requirements

- Python 3.7+
- Required dependencies (see `requirements.txt`)

## Getting Started

1. Download the dataset from the provided link
2. Install required dependencies
3. Configure parameters in the `config/` directory
4. Choose your execution mode (complete pipeline or individual modules)
5. Run the analysis

For detailed module-specific instructions, consult the individual README files in each module directory.
