# ECG_Beat_Classification

## 1. Introduction 
ECG plays an important role for diagnosis of cardiac diseases. However, pretrained models on such one-dimensional problems are scarce. 
This project strives to build a deep-learning-powered application for classify the morphology of heartbeats, which facilitate further downstream diagnosis tasks.

## 2. Technical Overview

### 2.1. Dataset
This project re-implements a part of the paper "ECG Heartbeat Classification: A Deep Transferable Representation" by Kachusee et al. By classify each heartbeat of dimension [1, 187] into five classes: [Normal beat, Supraventricular premature beat, Premature ventricular contraction, Fusion of ventricular and normal beat, Unclassifiable beat], we can enable transfer learning to a downstream task like miocardio infraction beat classification.

The MITBIH dataset have 109446 data entries (87554 train/ 21892 test). Each samples contains an ECG beat's morphology represented with 187 data points. The last item in each entry is the category of that beat.
There are five categories:

| Condition                                 | Class      | #Samples in Training Set | #Sample in Test set |
| :--------------------------------------   | :--------: | -----------------------: | ------------------: |
| Normal beat (N)                           | 0          | 72471                    | 18118               | 
| Supraventricular premature beat (S)       | 1          | 2223                     | 556                 | 
| Premature ventricular contraction (P)     | 2          | 5788                     | 1448                |
| Fusion of ventricular and normal beat (F) | 3          | 641                      | 162                 |
| Unclassifiable beat (U)                   | 4          | 6431                     | 1608                |

### 2.2. Model Architecture
The architecture is implemented as the author's suggestion:
```
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv1d-1              [-1, 32, 187]             192
                Conv1d-2              [-1, 32, 187]           5,152
                  ReLU-3              [-1, 32, 187]               0
                Conv1d-4              [-1, 32, 187]           5,152
                  ReLU-5              [-1, 32, 187]               0
            MaxPool1d-6               [-1, 32, 92]               0
        ResidualBlock-7               [-1, 32, 92]               0
                Conv1d-8               [-1, 32, 92]           5,152
                  ReLU-9               [-1, 32, 92]               0
              Conv1d-10               [-1, 32, 92]           5,152
                ReLU-11               [-1, 32, 92]               0
            MaxPool1d-12               [-1, 32, 44]               0
        ResidualBlock-13               [-1, 32, 44]               0
              Conv1d-14               [-1, 32, 44]           5,152
                ReLU-15               [-1, 32, 44]               0
              Conv1d-16               [-1, 32, 44]           5,152
                ReLU-17               [-1, 32, 44]               0
            MaxPool1d-18               [-1, 32, 20]               0
        ResidualBlock-19               [-1, 32, 20]               0
              Conv1d-20               [-1, 32, 20]           5,152
                ReLU-21               [-1, 32, 20]               0
              Conv1d-22               [-1, 32, 20]           5,152
                ReLU-23               [-1, 32, 20]               0
            MaxPool1d-24                [-1, 32, 8]               0
        ResidualBlock-25                [-1, 32, 8]               0
              Conv1d-26                [-1, 32, 8]           5,152
                ReLU-27                [-1, 32, 8]               0
              Conv1d-28                [-1, 32, 8]           5,152
                ReLU-29                [-1, 32, 8]               0
            MaxPool1d-30                [-1, 32, 2]               0
        ResidualBlock-31                [-1, 32, 2]               0
              Flatten-32                   [-1, 64]               0
              Linear-33                   [-1, 32]           2,080
                ReLU-34                   [-1, 32]               0
              Linear-35                   [-1, 32]           1,056
              Linear-36                    [-1, 5]             165
    ================================================================
    Total params: 55,013
    Trainable params: 55,013
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.47
    Params size (MB): 0.21
    Estimated Total Size (MB): 0.68
    ----------------------------------------------------------------
```
### 2.3. Result
The model archieve 97.265% accuracy on training set, and 97.067% accuracy on test set.

Confusion matrix:
```
          0         1         2         3         4
0  0.994149  0.000883  0.003643  0.000386  0.000938
1  0.437050  0.528777  0.028777  0.000000  0.005396
2  0.058011  0.000000  0.937155  0.002762  0.002072
3  0.432099  0.000000  0.166667  0.401235  0.000000
4  0.044776  0.000000  0.008706  0.000000  0.946517
```

## 3. Installation Guide
### Create virtual environment
```
    conda create -n ecg_env python=3.9
    conda activate ecg_env
```

### Install dependencies
```
    pip install -r requirements.txt
```

### Download data and unzip in folder <code>data</code>
```
    mkdir data
```
Dataset link: https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download

## 4. Usage section.
### Train models
Modify <code> config.json </code> file in <code> config </code> folder. Some samples are prepared in <code> samples </code> directory for prediction illustration. 

Train model
```
  python train.py
```

Test model
```
  python test.py
```

Predict on a sample
```
  python predict.py
```

See metrics on tensorboard
```
  tensorboard --logdir=log
```

### Run web apps for heartbeat visualization and inference.
Run web app:
```
  streamlit run app.py
```

## 5. Reference links
1. Paper link:
https://arxiv.org/pdf/1805.00794.pdf

2. Kaggle dataset:
https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download



