# DREAM for Combating Concept Drift

Official code for our [CCS'25 paper](https://arxiv.org/pdf/2405.04095) "Combating Concept Drift with Explanatory Detection and Adaptation for Android Malware Classification". 

## Setup

``` python
conda create --name dream python=3.8
conda activate dream
conda install conda-forge::tensorflow-gpu=2.11.0 # for CUDA 12.5
conda install -c conda-forge libgcc-ng libstdcxx-ng
conda install pandas matplotlib scikit-learn tqdm --yes
```
