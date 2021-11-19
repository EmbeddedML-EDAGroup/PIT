# Dataset

## PPG DaLiA
### Download
The dataset is freely available at https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA. 
Before running experiments on this dataset please download the data and update properly the path specified in **../config/config_Dalia.json**.

### Overview
PPG-DaLiA is a publicly available dataset for PPG-based heart rate estimation. This multimodal dataset features physiological and motion data, recorded from both a wrist- and a chest-worn device, of 15 subjects while performing a wide range of activities under close to real-life conditions. The included ECG data provides heart rate ground truth. The included PPG- and 3D-accelerometer data can be used for heart rate estimation, while compensating for motion artefacts. Further details can be found in the dataset's readme-file.

## ECG5000
### Download
The dataset is freely available at http://www.timeseriesclassification.com/description.php?Dataset=ECG5000.
Before running experiments on this dataset please download the data and update properly the path specified in **../config/config_ECG5000.json**.

### Overview
The original dataset for "ECG5000" is a 20-hour long ECG downloaded from Physionet. The name is BIDMC Congestive Heart Failure Database(chfdb) and it is record "chf07". It was originally published in "Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23)". The data was pre-processed in two steps: (1) extract each heartbeat, (2) make each heartbeat equal length using interpolation. This dataset was originally used in paper "A general framework for never-ending learning from time series streams", DAMI 29(6). After that, 5,000 heartbeats were randomly selected. The patient has severe congestive heart failure and the class values were obtained by automated annotation.

## NinaProDB1
### Download
The dataset is available at http://ninapro.hevs.ch/ after proper registration on the website.
Before running experiments on this dataset please download the data and update properly the path specified in **../config/config_NinaProDB1.json**.

### Overview
The first Ninapro database includes 27 intact subjects acquired with the acquisition protocol described into the papers: "Manfredo Atzori, Arjan Gijsberts, Ilja Kuzborskij, Simone Heynen, Anne-Gabrielle Mittaz Hager, Olivier Deriaz, Claudio Castellini, Henning Müller and Barbara Caputo. Characterization of a Benchmark Database for Myoelectric Movement Classification. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2014" (http://publications.hevs.ch/index.php/publications/show/1715) and "Manfredo Atzori, Arjan Gijsberts, Claudio Castellini, Barbara Caputo, Anne-Gabrielle Mittaz Hager, Simone Elsig, Giorgio Giatsidis, Franco Bassetto & Henning Müller. Electromyography data for non-invasive naturally-controlled robotic hand prostheses. Scientific Data, 2014" (http://www.nature.com/articles/sdata201453).

## Google Speech Commands
### Download
The pre-processed data are obtained using the scripts available in the TinyML-perf repository (https://github.com/mlcommons/tiny/tree/0b04bcd402ee28f84e79fa86d8bb8e731d9497b8/v0.5/training/keyword_spotting). Please run and save the loaded data as **pkl** files and update properly the path specified in **../config/config_GoogleSpeechCommands.json**.

### Overview
An audio dataset of spoken words designed to help train and evaluate keyword spotting systems. Its primary goal is to provide a way to build and test small models that detect when a single word is spoken, from a set of ten target words, with as few false positives as possible from background noise or unrelated speech. Note that in the train and validation set, the label "unknown" is much more prevalent than the labels of the target words or background noise. One difference from the release version is the handling of silent segments. While in the test set the silence segments are regular 1 second files, in the training they are provided as long segments under "background_noise" folder. Here we split these background noise into 1 second clips, and also keep one of the files for the validation set.