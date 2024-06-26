# Pencil 
**Pencil: Private and Extensible Collaborative Learning without the Non-Colluding Assumption**

Pencil is published in NDSS 2024 =>
[[Paper link]](https://www.ndss-symposium.org/ndss-paper/pencil-private-and-extensible-collaborative-learning-without-the-non-colluding-assumption/)

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
* openssl
* cmake 3.27.2
* CUDA 11.7 and a supporting GPU device
* python 3.8 or higher **with dev** (e.g. python3.8-dev) 
* python packages `cmake numpy pickle torch torchvision torchtext transformers`

### Building

1. Building the dependencies.

    For ease of use, just run
    ```bash
    bash scripts/build_dependencies.sh
    ```

    If you encounter any building problems, you could try building the three dependencies individually to locate the problem.
    ```bash
    bash scripts/build_opencheetah.sh
    bash scripts/build_seal_cuda.sh
    bash scripts/build_ezpc.sh
    ```

2. Install `libtroy.so`.

    After building the dependencies, this shared library can be found in `./seal-cuda/build/src/libtroy.so`. 

    If you have root privilige, you could copy it or create its soft link in `/usr/local/lib` or `/usr/lib`; otherwise, you can put it in any directory and add the directory to `LD_LIBRARY_PATH` by running `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/the/file` (This is temporary; for permanant change, you could add it to your `~/.bashrc` file).

### Frequently asked questions

* `#include<Python.h>` no such file or directory. 

    This is due to python-dev not found. Compiling the python module for `seal-cuda` explicitly requires the user to specify the python header files in `seal-cuda/makepackage.sh` Line 22. If you are using a virtual environment or conda, etc., you could specify the path as the corresponding python location, e.g. `.../anaconda3/pkgs/python-3.8.../include/python3.8/` (where you can find Python.h file).

* ImportError: libtroy.so: cannot open shared object file: No such file or directory

    This is because the linker cannot find `libtroy.so` shared library when running the scripts. Please refer to (2) "Install `libtroy.so`" for the solution.

* Could not find a package configuration file provided by "emp-tool"

    Normally compiling `OpenCheetah` would automatically find `emp-tool` and `emp-ot` in the build directory. If this error occurs, we recommend removing the `build` directory in`OpenCheetah` and rebuild from scratch. Alternatively, you could manually install `emp-ot` and `emp-tool` from the original repositories. ([emp-tool](https://github.com/emp-toolkit/emp-tool), [emp-ot](https://github.com/emp-toolkit/emp-ot))

* Address already in use

    If the training scripts are terminated by the user or by exception, the opened port might not be released correctly. The system will shut the sockets down after a while, but if you wish to immediately relaunch the training process, you could edit the port number in `config.py` to use some other port.

### Google Colab Notebook

To simply check the functionalities, you could alternatively use the [Google Colab Notebook](https://colab.research.google.com/drive/12O80pXUsS4aPipCrRrU9k2j77m0crO7P?usp=sharing) we prepared. Before running, please unzip `library-files.zip` in the repository and upload them to the default directory (`/content`) of colab.

## Artifact evaluation

### Basic usage

The main python script for running pencil is the `train_priv_server.py` and `train_priv_client.py` files in `pencil-fullhe` and `pencil-prep` folder. These correspond to the two versions of Pencil without or with the preprocessing optimization technique.

```bash
cd pencil-fullhe # or pencil-prep

# run in two different terminals
python train_priv_server.py
python train_priv_client.py
```

For simple test of functionality, one can run the `scripts/fullhe-trivial.sh` or `scripts/prep-trivial.sh`, which will run Pencil on MNIST/NN1 for 1 epoch. The logs will be spawned in the corresponding `pencil-fullhe/logs` or `pencil-prep/logs` folder.

### End-to-end training

`train_priv_server.py` accepts arguments:

* `-p` preset, which may be `mnist_aby3, mnist_chameleon, agnews_cnn, cifar10_sphinx, agnews_gpt2, resnet50_classifier` which corresponds the six NNs as listed in the paper. There are some other presets, which is listed in `model.py` function `get_model`.
* `-n` DP noise levels added to the gradients. This corresponds to $\sigma$ in Equation 5 and Table IV.
* `-C` estimated gradient bound. This corresponds to $C$ in Equation 5.
* `-o` optimizer. Supported are `SGD, Adam`.
* `-om` optimizer momentum. Only available for `SGD` optimizer.
* `-lr` learning rate.

`train_priv_client.py` accepts arguments:

* `-s` how to split the data as different DOes. 
    * `None`: only one DO.
    * `p:q` (where `p` and `q` are integers): (1) take `q/p` of data as set `S` and the rest as `T`; (2) Sort `S` by labels, `T` is unchanged; (3) `S` and `T` are both sliced into `p` parts with the same amount of data, and `p` DOes each take one part from `S` and `T`. *For example*: `p=2, q=1` for MNIST with 60000 pictures and 10 labels. There will be two DOes `A` and `B`, where `A` has 15000 pictures only with labels `0~4` and 15000 pictures of all 10 labels, and `B` has 15000 pictures only with labels `5~9` and 15000 pictures of all 10 labels.
* `-e` epochs.

### Training cost measuring

Use `server_costs.py` and `client_costs.py` for measuring the time and communication cost for training one step for different models. The `server_costs.py` accepts the preset `-p` argument like above.

### Artifact evaluation scripts

`scripts/gen_scripts.py` could be used to generate a series of evaluation scripts, which could be used to reproduce the main results presented in the paper. They are all prefixed with `prep_` or `fullhe_` for the two variations of Pencil. These scripts will generate log files in the corresponding `logs/` folder. You could look at the contents of the scripts for further details of how these scripts invoke the underlying python files.

* `train_nn*.sh`: Training end to end. The client output log contains the accuracy results.
* `costs_nn*.sh`: Measure the training cost for one step in the training. The server output log contains step communication and time outputs, and for `prep` variation also contains preparation time and communication results.
* `dp*_nn*.sh`: Training end to end with DP noises added. These results corresponds to Table IV in the paper.
* `hetero_nn*.sh`: Train with heterogeneous DOes. These results corresponds to Fig. 2 in the paper.

