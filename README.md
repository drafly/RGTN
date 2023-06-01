# RGTN

Code for the paper [Region Guided Transformer for Single Image Raindrop Removal]

### About this repo:

This repo hosts the implentation code for the paper "RGTN". 

## Introduction

Image quality can be significantly affected by the presence of raindrops, resulting in unwanted reflections and occlusions. While methods based on deep learning have shown great potential in removing these artifacts, they often struggle to completely remove the traces of the raindrops and restore the original scene. To overcome this limitation, we propose a novel Region Guided Transformer Network(RGTN) for single image raindrop removal. The proposed RGTN is composed of a unique attention mechanism called Mask-Window Multihead Self-Attention (MW-MSA), which uses a degradation region mask to focus on clean background information by differentially processing degraded regions. Comprehensive experiments show that our RGTN network is superior to the existing methods on raindrop removal benchmark datasets, producing more detailed and realistic results. To further illustrate the demonstrate the generalization ability and robustness of our network, we also conducted experiments using datasets from other related tasks, such as snow removal, and our network gains superior performance compared to the latest methods on Snow 100K dataset. 

## Network Architecture

<img src = "https://github.com/converT98/RGTN/blob/main/images/network.png"> 

## Using the code:

The code is stable while using Python 3.8.11, CUDA >=11.3

- Clone this repository:
```bash
git clone (https://github.com/converT98/RGTN.git)
cd RGTN
```

- To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate RGTN
```
- Quick start：
```bash
python train.py
%modifiy your path in Options/basic.yml
```

## Datasets:

### Train Data:

RGTN is trained on Raindrop datasets and Snow100K

[Raindrop](https://rui1996.github.io/raindrop/raindrop_removal.html)
[Snow100K](https://sites.google.com/view/yunfuliu/desnownet)

### Acknowledgements:


### Citation:

```

```
