# Exoplanet Detection 🪐

## Introduction
A Deep Learning pipeline combining Convolutional Neural Networks, Dimensionality Reduction and Classifiers to perform a multi-class classification of NASA's Kepler and TESS Threshold-Crossing Events.

**NOTE**. Currently, any use of the code for publication purposes is prohibited without my consent (Stefano Fiscale).

---

## Dataset

**Download Link:** https://drive.google.com/file/d/18rUMHXPRWOwFYCNCZ-ph27aB-qbEPUtO/view?usp=sharing

## Pipeline Execution

1. **Dataset Preprocessing** - Creating, loading, training-test split
2. **Feature Extraction** - Identification and extraction of relevant astronomical features
3. **Dimensionality Reduction** - Optimization of feature space for improved performance
4. **Classification** - Model training and exoplanet classification

## Execution Modes

To ensure code reusability and adaptability for other projects, the system is designed with independent, modular components that can be combined to create a complete ML pipeline.

Two execution modes are available:
* Individual Modules: Execute specific components independently for targeted analysis or debugging.
*  Complete Pipeline: Automated execution of all modules in sequence for end-to-end processing.

## Configuration Files

Configuration files for individual modules are located in the `config/` directory. These files enable execution with custom parameters tailored to specific requirements.

For detailed parameter descriptions and usage instructions, refer to the respective `readme_<module_name>.md` files in each module directory.

**Note:** Parameters are loaded by individual modules and are also applied during Complete Pipeline execution.

## Usage
Di seguito forniamo un esempio di utilizzo di ognuno dei moduli sviluppati.

### 1. Feature Extraction with Convolutional Neural Network (CNN)
This module allows you to train and test a Convolutional Neural Network (CNN) in order to extract the feature vectors from the last feature extraction layer of the model.
It is also possible to use the trained CNN as a classifier by setting it to inference mode.

Configuration files to define input parameters:
-- config_dataset.yaml
-- config_feature_extractor.yaml
-- config_resnet.yaml (or config_vgg.yaml), depending on the CNN architecture you wish to use as the feature extractor)

```bash
conda activate <your_env_name>
python3 main/code/feature_extraction/feature_extractor.py > main/output_files/<name_of_output_file>.out
```

After execution, the following files will be generated:

- [data/features_step1_cnn/]: Two .npy files containing the extracted feature vectors and the corresponding labels of the processed signals.
- [output_files/]: A .out file listing all the operations performed during the execution.
- [output_files/training_metrics/feature_extractor/]: Five .png files showing the evolution of evaluation metrics (Loss, AUC, Precision, Recall, F1-score) on the training and validation sets (if used).


### 2. 

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
