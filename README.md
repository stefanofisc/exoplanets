# Exoplanets detection with Deep Learning

## Introduction


## How to run the code
Per adesso, il codice può essere eseguito nel seguente modo:

1. In ```[main/code/dataset/config_dataset.yaml]```  Modifica le variabili 'dataset_filename' e 'catalog_name' per caricare uno dei tre dataset in formato PyTorch TensorDataset. Il codice caricherà gli split training-test (80%-20%) che ho preventivamente generato per ogni dataset.
2. In ```[main/code/feature_extraction/config_feature_extractor.yaml]``` Seleziona il nome della Convolutional Neural Network che vuoi usare come estrattore di caratteristiche: vgg o resnet.
3. In ```[feature_extraction/vgg (o resnet)/]``` Vai nel relativo file di configurazione (e.g. ```config_vgg.yaml```) per settare i parametri architetturali e di training del modello.
4. Da riga di comando, esegui:

```bash
python3 feature_extractor.py > ../../output_files/<YYYY-MM-DD>_<model>_<further_information>.out
```

Per esempio:
```bash
python3 feature_extractor.py > ../../output_files/2025-06-02_feature_extractor_test-6_integrating_Dataset-VGG-Resnet-Model.out
```


### Setting up conda environment


## Data

Link to download the dataset: https://drive.google.com/file/d/18rUMHXPRWOwFYCNCZ-ph27aB-qbEPUtO/view?usp=sharing


## Development
Link to the Google Document we are using to keep track about any update: https://docs.google.com/document/d/1InzZJqh13LRLW6fWPuhjtBzl0_MMm5-JdZ-Xc6xOX6w/edit?usp=sharing
