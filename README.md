# models
* letnet 264KB
* ResNet18 43MB

# training
- [x] resnet18 vanilla: 94% (teacher)
- [x] resnet18 mixup: 95%
- [x] lenet vanilla: 66%
- [x] lenet kd: 75.5%
- [x] lenet kd + temperature2 (kd_temp2): 75%
- [x] lenet kd + mixup (alpha=0.2): 76%
- [x] lenet kd + mixup (alpha=1): 75%
- [x] lenet kd + mixup (alpha=2): 75%
- [x] lenet kd + manifold mixup (alpha=0.2): 76%
- [ ] lenet kd + manifold mixup (alpha=1):
- [ ] lenet kd + fitnet + manifold mixup
- [ ] 


# Mixup-CIFAR10
By [Hongyi Zhang](http://web.mit.edu/~hongyiz/www/), [Moustapha Cisse](https://mine.kaust.edu.sa/Pages/cisse.aspx), [Yann Dauphin](http://dauphin.io/), [David Lopez-Paz](https://lopezpaz.org/).

Facebook AI Research

## Introduction

Mixup is a generic and straightforward data augmentation principle.
In essence, mixup trains a neural network on convex combinations of pairs of
examples and their labels. By doing so, mixup regularizes the neural network to
favor simple linear behavior in-between training examples.

This repository contains the implementation used for the results in
our paper (https://arxiv.org/abs/1710.09412).

## Citation

If you use this method or this code in your paper, then please cite it:

```
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
```

## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20170922 --decay=1e-4
```

## License

This project is CC-BY-NC-licensed.

## Acknowledgement
The CIFAR-10 reimplementation of _mixup_ is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).
