#!/bin/bash

#SBATCH --job-name=graph_plotting
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=graphics.out
#SBATCH --error=graphics.err

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate PGx_env

# Install libs
#pip install seaborn 
#pip install matplotlib 

# Plot the graphs
python graphics.py