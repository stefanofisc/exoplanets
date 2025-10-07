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

### Complete Pipeline Execution
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
