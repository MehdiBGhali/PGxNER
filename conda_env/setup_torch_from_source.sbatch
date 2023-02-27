#!/bin/bash

#SBATCH --job-name=torch_install_from_source
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=torch_install.out
#SBATCH --error=torch_install.err

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate final_PGx_env_latest_transformers

# Check for CUDA
conda list cudatoolkit
conda install -c "nvidia/label/cuda-11.7" cuda-nvcc

# Uninstall previous torch version
pip uninstall torch -y
conda uninstall torch -y
# Install torch
conda install cmake ninja
rm -r pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

