# Complement Objective Training


## Overview

This repository contains the PyTorch implementation of Complement Objective Training introduced in the following paper:

> _COMPLEMENT OBJECTIVE TRAINING_. <br>
**Hao-Yun Chen**, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, Da-Cheng Juan. <br> <https://openreview.net/forum?id=HyM7AiA5YX>

## Introduction
Complement Objective Training is a new training paradigm that updates neural network parameters by alternating iteratively between the primary objective and the complement objective. Conventionally, training with cross entropy as the primary objective aims at maximizing the predicted probability of the ground-truth class, while we propose “complement entropy” as the complement objective for neutralizing the predicted probabilities of the complement classes. The experimental results confirm that, compared to the conventional training with just one primary objective, training also with the complement objective further improves the performance of the state-of-the-art models across all tasks.

## Dependencies

* Python 3.6 
* Pytorch 0.4.1


## Usage
For getting baseline results
	
	python main.py --sess Baseline_session
	
For training via Complement objective

	python main.py --COT --sess COT_session


## Benchmark on CIFAR10

The following table shows the best test errors in a 200-epoch training session. (Please refer to Figure 3a in the paper for details.)

| Model              | Baseline  | COT |
|:-------------------|:---------------------|:---------------------|
| PreAct ResNet-18                |               5.46%  |               4.86%  |


## Citation
If you find this work useful in your research, please cite:
```bash
@inproceedings{
chen2018complement,
title={Complement Objective Training},
author={Hao-Yun Chen and Pei-Hsin Wang and Chun-Hao Liu and Shih-Chieh Chang and Jia-Yu Pan and Yu-Ting Chen and Wei Wei and Da-Cheng Juan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=HyM7AiA5YX},
}
```

## Acknowledgement
The CIFAR-10 reimplementation of COT is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).

