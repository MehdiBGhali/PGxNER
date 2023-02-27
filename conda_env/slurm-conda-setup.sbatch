#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Create the environment using python 3.8
conda create --name final_PGx_env python=3.8 --force
# Create the environment using python 3.8
#conda create --name clean_PGx_env_latest_transformers python=3.8 --force

# Activate the environment
source activate final_PGx_env
# Install the required libraries from a requirement file, using pypi
pip install -r requirements_no_torch.txt

# Activate the environment
#source activate clean_PGx_env_latest_transformers
# Install the required libraries from a requirement file, using pypi
#pip install -r requirements_no_torch.txt 