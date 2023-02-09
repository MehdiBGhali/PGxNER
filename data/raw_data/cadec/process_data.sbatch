#!/bin/bash

#SBATCH --job-name=Build_data_from_cadec_dataset
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate PGx_env

# Create storage directory
mkdir ./processed_data

# Go to processing tools directory
cd ../../PGxNER/processing_tools_from_acl2020/cadec

# Extract annotations from original dataset
echo "Extract annotations ..." >> ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log
touch ../../../data/raw_data/cadec/processed_data/ann # Output directory
python extract_annotations.py --input_text ../../../data/raw_data/cadec/text --input_ann ../../../data/raw_data/cadec/original --output_filepath ../../../data/raw_data/cadec/processed_data/ann --type_of_interest ADR --log_filepath ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log

# Tokenize data
echo "Tokenization ..." >> ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log
touch ../../../data/raw_data/cadec/processed_data/tokens # Output directory
python tokenization.py --input_dir ../../../data/raw_data/cadec/text --output_filepath ../../../data/raw_data/cadec/processed_data/tokens --log_filepath ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log

# Convert annotations from character level offsets to token level idx
echo "Convert annotations from character level offsets to token level idx ..." >> ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log
touch ../../../data/raw_data/cadec/processed_data/tokens.ann # Output directory
python convert_ann_using_token_idx.py --input_ann ../../../data/raw_data/cadec/processed_data/ann --output_ann ../../../data/raw_data/cadec/processed_data/tokens.ann --log_filepath ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log

# Put data in algorithm's standard format
echo "Create text inline format ..." >> ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log
touch ../../../data/raw_data/cadec/processed_data/inline # Output directory
python convert_text_inline.py --input_ann ../../../data/raw_data/cadec/processed_data/tokens.ann --output_filepath ../../../data/raw_data/cadec/processed_data/inline --log_filepath ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log

# Train/test/dev split
echo "Split the data set into train, dev, test splits ..." >> ../../../data/raw_data/cadec/build_data_for_transition_discontinous_ner.log
mkdir ../../../data/raw_data/cadec/processed_data/split # Output directory
python split_train_test.py --input_filepath ../../../data/raw_data/cadec/processed_data/inline --output_dir ../../../data/raw_data/cadec/processed_data/split