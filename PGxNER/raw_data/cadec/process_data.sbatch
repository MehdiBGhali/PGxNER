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
cd ../processing_tools_from_acl2020/cadec

# Extract annotations from original dataset
echo "Extract annotations ..." >> ../../cadec/build_data_for_transition_discontinous_ner.log
touch ../../cadec/processed_data/ann # Output directory
python extract_annotations.py --input_text ../../cadec/text --input_ann ../../cadec/original --output_filepath ../../cadec/processed_data/ann --type_of_interest ADR --log_filepath ../../cadec/build_data_for_transition_discontinous_ner.log

# Tokenize data
echo "Tokenization ..." >> ../../cadec/build_data_for_transition_discontinous_ner.log
touch ../../cadec/processed_data/tokens # Output directory
python tokenization.py --input_dir ../../cadec/text --output_filepath ../../cadec/processed_data/tokens --log_filepath ../../cadec/build_data_for_transition_discontinous_ner.log

# Convert annotations from character level offsets to token level idx
echo "Convert annotations from character level offsets to token level idx ..." >> ../../cadec/build_data_for_transition_discontinous_ner.log
touch ../../cadec/processed_data/tokens.ann # Output directory
python convert_ann_using_token_idx.py --input_ann ../../cadec/processed_data/ann --output_ann ../../cadec/processed_data/tokens.ann --log_filepath ../../cadec/build_data_for_transition_discontinous_ner.log

# Put data in algorithm's standard format
echo "Create text inline format ..." >> ../../cadec/build_data_for_transition_discontinous_ner.log
touch ../../cadec/processed_data/inline # Output directory
python convert_text_inline.py --input_ann ../../cadec/processed_data/tokens.ann --output_filepath ../../cadec/processed_data/inline --log_filepath ../../cadec/build_data_for_transition_discontinous_ner.log

# Train/test/dev split
echo "Split the data set into train, dev, test splits ..." >> ../../cadec/build_data_for_transition_discontinous_ner.log
mkdir ../../cadec/processed_data/split # Output directory
python split_train_test.py --input_filepath ../../cadec/processed_data/inline --output_dir ../../cadec/processed_data/split