# Pencil 
**Private and extensible collaborative NN training framework**

## Overview

As a private NN training framework, Pencil uses EzPC/OpenCheetah for the non-linearities and GPU-assisted HE (seal-cuda) for linear evaluations.
Therefore, the repository has
* [EzPC](https://github.com/mpc-msri/EzPC) submodule, [edited](https://github.com/lightbulb128/EzPC) to provide python utilities of 2PC sqrt, elementwise multiplication, etc.
* [OpenCheetah](https://github.com/Alibaba-Gemini-Lab/OpenCheetah) submodule, [edited](https://github.com/lightbulb128/OpenCheetah) to provide python utilities of ReLU, DRelu, etc.
* [seal-cuda](https://github.com/lightbulb128/troy) submodule, a fully functional GPU-assisted homomorphic encryption library.
* pencil-fullhe, the version of Pencil without the preprocessing technique.
* pencil-prep, the version of Pencil with the preprocessing technique.


## Build and run

### Requirements
* g++ 11.4.0
* python 3.8
* cmake 3.27.2
* CUDA 11.7 and a supporting GPU device

### Building

For ease of use, juse run
```bash
bash scripts/build_dependencies
```
This requires `sudo` privilige as it needs to install SEAL and GPU assisted utilities.

### Running
```bash
cd pencil-fullhe # or pencil-prep

# run in two different terminals
python train_priv_server.py
python train_priv_client.py
```