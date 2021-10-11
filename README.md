# Deep Generative Learning via Euler Particle Transport

This repository is the official implementation of [Deep Generative Learning via Euler Particle Transport]. 

## Introduction
The **Euler Particle Transport approach (EPT)** is proposed for deep generative learning with theoretical guarantees by integrating ideas from optimal transport, numerical ODEs,  density-ratio (density-difference) estimation, and deep neural networks. It is evaluated on both simulated data and real image data.

## Dependencies
* Python 3.7.7
* PyTorch 1.4.0
* torchvision 0.5.0

## Experimental Results

### 2D Examples

To run EPT with the density-ratio based energy functional on 2D simulated data, use:

```
python 2D_lsdr.py --data_source <name_of_source_data> --data_target <name_of_target_data>
```

To run EPT with the density-difference based energy functional on 2D simulated data, use:

```
python 2D_dd.py --data_source <name_of_source_data> --data_target <name_of_target_data>
```

### Benchmark Image Datasets

To run EPT without the outer loop on CIFAR-10, use:

```
python main_lsdr_no_ol.py -div Pearson --dataset cifar10 --dataroot <data_path> --net resnet
```

To run EPT with the outer loop on CIFAR-10, use:

```
python main_lsdr_ol.py -div Pearson --dataset cifar10 --dataroot <data_path> --net resnet
```
