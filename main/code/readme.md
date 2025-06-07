### EN
This file describes how dataset and feature_extraction modules work.

The model processes as input phase-folded and binned light curves, which represent the transits to be classified. These light curves are stored in csv-format files, and each is associated with a label, which identifies its nature (e.g. planet, astrophysical false positive, non-astrophysical false positive). 

The dataset folder contains the code responsible for processing the light curves. Through the configuration file config_dataset.yaml it is possible to choose the dataset to be processed, and to perform a number of operations such as generating/loading train-test splits in csv or tensor format. Since the model processes data in torch.utils.data.DataLoader format, we suggest directly loading data in tensor format by setting the variable load_tensors:True. This lightens the computational load of the entire pipeline.

Once loaded, the light curves are processed through the feature_extraction module. This module is responsible for feature extraction from the data, a process that is carried out by one of the three Convolutional Neural Networks (CNNs) we currently provide: VGG-19, Resnet-18, Resnet-34. The features are extracted and saved during the last training epoch of the CNN. The hyperparameters of the training are set in the config_feature_extractor.yaml file, while the architectural hyperparameters of the model are defined in config_<model_name>.yaml. At the end of the network training, the features are stored in main/data/features_step1_cnn/, separately from their respective labels, but in two .npy files that follow the following standard in name: 

``<YYYY-MM-DD>_<model_name>_<optimizer>_<num_epochs>_<dataset_name>_features.npy`` (or _labels.npy).

An output file, in .out format, reporting the training details is saved in main/output_files/, while plots of the training metrics are saved in main/output_files/training_metrics/. The names of the output files and plots follow the same format as those for the saved features.

### IT
Questo file descrive il funzionamento dei moduli dataset e feature_extraction.

Il modello processa in input curve di luce 'phase-folded' e binnate, che rappresentano i transiti da classificare. Queste curve di luce sono memorizzate in file di formato csv, e ad ognuna di esse è associata un'etichetta, che ne identifica la natura (e.g. pianeta, falso positivo astrofisico, falso positivo non-astrofisico). 

La cartella dataset contiene il codice responsabile del processamento delle curve di luce. Attraverso il file di configurazione config_dataset.yaml è possibile scegliere il dataset da processare, ed effettuare una serie di operazioni quali: generazione / caricamento di train-test splits in formato csv o tensoriale. Siccome il modello processa dati in formato torch.utils.data.DataLoader, suggeriamo di caricare direttamente i dati in formato tensoriale, settando la variabile load_tensors:True. Questo alleggerisce il carico computazionale dell'intera pipeline.

Una volta caricate, le curve di luce vengono processate attraverso il modulo feature_extraction. Questo modulo è responsabile dell'estrazione di caratteristiche dai dati, un processo che viene effettuato da una delle tre Convolutional Neural Networks (CNNs) che attualmente mettiamo a disposizione: VGG-19, Resnet-18, Resnet-34. Le caratteristiche vengono estratte e salvate durante l'ultima epoca di training della CNN. Gli iperparametri del training vengono settati nel file config_feature_extractor.yaml, mentre gli iperparametri architetturali del modello sono definiti in config_<model_name>.yaml. Al termine del training della rete, le features sono salvete in main/data/features_step1_cnn/, separatamente dalle rispettive etichette, ma in due file di tipo .npy che seguono il seguente standard nel nome: 

```<YYYY-MM-DD>_<model_name>_<optimizer>_<num_epochs>_<dataset_name>_features.npy``` (o _labels.npy).

Un file di output, in formato .out, che riporta i dettagli del training è salvato in main/output_files/, mentre i plot delle metriche di training sono salvati in main/output_files/training_metrics/. I nomi dei file di output e dei plot seguono lo stesso formato di quelli relativi alle caratteristiche salvate.
