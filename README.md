# Multi-view Deep Learning Framework for Precise Prediction of Transcription Factor Binding Sites

## Table of Contents
1. [Introduction](#Introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [Implementary](#Implementary)
## 1. Introduction
In this study, we propose a novel multi-view deep learning framework called Multi-view Deep Learning for Transcription Factor Binding Prediction (MDNet-TFP). Our framework introduces a bidirectional reverse complement module (BiRC-Mamba), which effectively accounts for the bidirectional and reverse complement properties characteristic of DNA sequences. Furthermore, we developed a multi-scale convolutional recurrent attention network (MCRAN) that extracts both structural and functional DNA features across multiple dimensions while integrating information from various biological datasets. In 165 CHIP-seq datasets, our model achieve average accuracy(ACC) of 0.881, ROC-AUC of 0.937 and PR-AUC of 0.934.

## 2. Python Environment
python XXX and packages version:

+ torch == XXXX
+ torchvision == XXX
+ mamba_ssm == XXX
+ numpy == XXX
+ matplotlib == XXX
+ scikit-learn == XXX
## 3. Project Structure
### 3.1 Dataset
- We choose the 165 ChIP-seq experimental datasets from the Encyclopedia of DNA Elements(ENCODE) database, which include TFBSs for 29 distinct TFs across 32 different cell lines. 
- We also use 690 ChIP-seq datasets to validate the generation of our model. You can get the dataset from [saresnet/data](https://csbioinformatics.njust.edu.cn/saresnet/html/Data.html)
### 3.2 Model

![Model Architecture](MDNet-TFP.png)
- ```dealwithdate.py```: Handles the preprocessing of DNA sequences for different model requirements. Supports SCE encoding, one-hot encoding, and reverse-complement transformation. The processed sequences are saved for downstream training and evaluation.
- ```MDNET-TFP.py```: Main training script that integrates features extracted by the MCRAN and BiRC-Mamba modules. Applies DFMFS (Dual-Feature Multi-scale Feature Selection) to select and combine informative features for classification tasks.
- ```birc_mamba.py```: Implements the BiRC-Mamba module, which incorporates a reverse complementary learning mechanism based on the Mamba architecture to enhance sequence representation.
- ```mcran.py```: Implements the MCRAN (Multi-scale Convolutional Residual Attention Network) module.
- ```mfss.py```: Implements the DFMFS (Dual-Feature Multi-scale Feature Selection) strategy, which adaptively fuses and selects features from both the MCRAN and BiRC-Mamba modules.

### 3.3 Implementary
- we can run the ```dealwithdate.py``` to process data.

    ```python dealwithdate.py```:
- To train and test the model, we can run the ```train.py```. In this process, we use three stage training (train MCRAN, BiRC-Mamba and DFMFS seperately).
  
    ```python train.py```:
