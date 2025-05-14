# plmc_pruned
This repository is based on code originally published under an MIT License by dpxiong (https://github.com/dpxiong/PLMC). This version is provided as-is with **no endorsement of the original implementation or associated claims**.

## PLMC modification
PLMC is a deep learning framework for protein crystallization prediction with protein language embeddings and handcrafted features. This code provides a version of PLMC that doesn't use protein language embeddings (for various reasons).

### Creating a conda environment
After cloning the repository, please run the following lines to install an environment for PLMC
```
conda create -n plmc_pruned python=3.9
conda activate plmc_pruned
conda install numpy=1.23.5
conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
conda install -c conda-forge scikit-learn=1.0.2
```

### Data preparation
Download the data at this [address](https://zenodo.org/record/6475529/), and uncompress it to the current directory.

### Training PLMC
Execute the following command:
```
python main.py --rawpath ./data/CRYS_DS
```

### Predicting crystallization outcome of your own protein:
**Is not possible (with neither this nor the original tool).**
