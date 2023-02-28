#!/bin/bash

#SBATCH --job-name=PGxCorpus_training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=logslurms/PGxCorpus/slurm-%j.out
#SBATCH --error=logslurms/PGxCorpus/slurm-%j.err

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate final_PGx_env

## Change Python version 
#conda install python=3.8
# Update pip before installing requirements
#pip install --upgrade pip
# Verify version
#python --version

## Install requirements
## conda install -c conda-forge rust
#rustc --version 
#pip install setuptools_rust # Required for tokenizers which are required for transformers
##pip install tokenizers # Required for transformers requirement
#pip install -r ./requirements_no_torch.txt
#pip install git+https://github.com/fastnlp/fastNLP@dev
#pip install git+https://github.com/fastnlp/fitlog

## Uninstall previous torch version
#pip uninstall torch -y
#conda uninstall torch -y
## Install torch
#conda install -c conda-forge clang 
#conda install -c anaconda cudatoolkit
#conda install mkl mkl-include
#conda install -c pytorch magma-cuda110
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#python setup.py develop

# Run training for PGxCorpus
python ./BARTNER_adapted/train.py --dataset_name PGxCorpus --history_dir ./training_history  --save_model 1 \
--n_epochs 100 --max_len 10 --max_len_a 1.6 --num_beams 3 --bart_name facebook/bart-large