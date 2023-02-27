#!/bin/bash

#SBATCH --job-name=Process_PGxCorpus
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=Process_PGxCorpus.out
#SBATCH --error=Process_PGxCorpus.err

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate PGx_env

# Create storage directory
# mkdir ./processed_data


# Split the data 
# pip install scikit-learn
# python split.py --test 0.2 --dev 0.2 # If first time running, move the split to processing tools

# Go to processing tools directory
cd ../../../PGxNER/processing_tools_from_acl2020/PGxCorpus

# Extract annotations from original dataset
echo "Extract annotations ..." >> ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log
touch ../../../data/raw_data/PGxCorpus/processed_data/ann # Output directory
python extract_annotations.py --input_text ../../../data/raw_data/PGxCorpus/text --input_ann ../../../data/raw_data/PGxCorpus/ann --output_filepath ../../../data/raw_data/PGxCorpus/processed_data/ann --log_filepath ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log

# Tokenize data
echo "Tokenization ..." >> ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log
touch ../../../data/raw_data/PGxCorpus/processed_data/tokens # Output directory
python tokenization.py --input_dir ../../../data/raw_data/PGxCorpus/text --output_filepath ../../../data/raw_data/PGxCorpus/processed_data/tokens --log_filepath ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log

# Convert annotations from character level offsets to token level idx
echo "Convert annotations from character level offsets to token level idx ..." >> ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log
touch ../../../data/raw_data/PGxCorpus/processed_data/tokens.ann # Output directory
python convert_ann_using_token_idx.py --input_tokens ../../../data/raw_data/PGxCorpus/processed_data/tokens --input_ann ../../../data/raw_data/PGxCorpus/processed_data/ann --output_ann ../../../data/raw_data/PGxCorpus/processed_data/tokens.ann --log_filepath ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log

# Put data in algorithm's standard format
echo "Create text inline format ..." >> ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log
touch ../../../data/raw_data/PGxCorpus/processed_data/inline # Output directory
python convert_text_inline.py --input_ann ../../../data/raw_data/PGxCorpus/processed_data/tokens.ann --input_tokens ../../../data/raw_data/PGxCorpus/processed_data/tokens --output_filepath ../../../data/raw_data/PGxCorpus/processed_data/inline --log_filepath ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log

# Train/test/dev split
echo "Split the data set into train, dev, test splits ..." >> ../../../data/raw_data/PGxCorpus/Process_PGxCorpus.log
# mkdir ../../../data/PGxCorpus # Output directory
python split_train_test.py --input_filepath ../../../data/raw_data/PGxCorpus/processed_data/inline --output_dir ../../../data/PGxCorpus