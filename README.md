# QNAS - PyTorch Version

## Introduction

## Convolutional Neural Network


## Environment Configuration

The following steps are used to configure the environment for the project.

- Miniconda Installation
- Conda Environment Creation
- Package Installation

**Notes**: 
- An NVIDIA GPU is required to run the project. 
- The project was tested using three NVIDIA RTX A30 GPUs to run evolutionary search with up to 20 individuals in parallel.
- NVIDIA drivers and the CUDA Toolkit are necessary (tested with CUDA 11.6).
- The following steps have been tested on Ubuntu 20.04 and WSL2 - Ubuntu 20.04.

### Miniconda Installation

Install Miniconda in the home directory. Refer to the [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) for more information.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Conda Environment Creation

```bash
conda create -n qnas python=3.9
conda activate qnas
```

### Package Installation

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
