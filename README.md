# DeepNm
## Introduction
2’-O-Methylation (2’-O-Me) is a post-transcriptional RNA modification that occurs in the ribose sugar moiety of all four nucleotides and is abundant in both coding and non-coding RNAs. Previous studies revealed its vital role in RNA metabolism and functions. High-throughput approaches such as Nm-seq have been invented to identify the 2’-O-Me sites with base-resolution, and multiple computational frameworks were also developed as useful alternatives. However, there is still a lack of computational tools to provide high-accuracy prediction and identify specific sequence motifs for each subtype of 2’-O-Me, i.e., Am, Cm, Gm, and Um. 

We first propose a novel deep learning model DeepNm to better capture the sequence features of each subtype with a multi-scale feature fusion framework. Based on DeepNm, we continue to propose HybridNm, which combines sequences and nanopore signals through a dual-path framework. Through validation on the benchmark dataset constructed from Nm-seq data, the newly proposed framework achieved promising performance. By incorporating additional Nanopore signal-derived features into the primary sequence model, significant improvement is observed on an already high baseline. Notably, through model interpretation, we not only identified subtype-specific motifs, but also observed shared motifs between subtypes. In addition, Cm, Gm, and Um shared motifs with the well-studied m6A RNA methylation.
## Requirements
- Python 3.x (3.8.8)
- Tensorflow 2.3.2
- Numpy 1.18.5
- Pandas 1.2.4
- scikit-learn 0.24.1
- prettytable 2.1.0 
## Installation
Please clone this repository as follows:
```
git clone https://github.com/daiyun02211/DeepNm.git
cd ./DeepNm
```
## Usage
Python codes for DeepNm and HybridNm can be found in Scripts/main.py:
```
python Scripts/main.py --mode eval --model DeepNm --name Am
```
Optional arguments are provided to ease usage:
- ``--mode``: Three modes can be selected: train, eval and infer;
- ``--model``: Two models can be selected: DeepNm and HybridNm;
- ``--name``: Four subtypes can be selected: Am, Cm, Gm, and Um
- ``--data_dir``: The directory where the processed data is stored;
- ``--cp_dir``: The directory where the trained network weights (checkpoints) are stored.
Further arguments can be found:
```
python Scripts/main.py -h
```