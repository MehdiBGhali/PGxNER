# PGxNER
A repo for Named Entity Recognition experiments on PGxCorpus data (hand-annotated pharmacogenetics corpus)

We are exploring a state-of-the-art solution to perform named entity recognition tasks on PGxCorpus, a manually annotated dataset of 
pharmacogenetic knowledge for natural language processing tasks

The current best performance is sitting at a 0.7169 f-score on mixed contiguous and discontinuous NER examples data, using a BART_large
based on the work from [Yan et al. (2021)](https://arxiv.org/abs/2106.01223)

# Execution Environment 

The [conda_env](/conda_env) directory contains scripts to recreate the conda environments used during this work in a SLURM remote environment. Run *slurm-conda-setup.bash* 
followed by a *requirements.txt* installation to create conda env. If CUDA/Torch errors appear, a Pytorch install from source might be required. In which case run
*setup_torch_from_source.sbatch*
followed by *requirements_no_torch.txt* instead

All the experiments were conducted under Python version==3.8  

# Preparing Data 

The [data](/data) directory contains the datasets used for the experiment. Each folder is named after the corresponding dataset, 
pre-formatted and split into three files: *train.txt*, *test.txt*, and *dev.txt*. A sub-folder called *raw_data* contains the raw data for each dataset, 
as well as scripts for generating the data in the proper format from these raw files.

The format to be followed for the data files is the following : two lines per sample, one empty line between two samples. The second 
line should be empty if the sentence does not contain any entities. Example : 

```
  Abdominal cramps, flatulence, gas, bloating. 
  0,1 ADR|3,3 ADR|7,7 ADR|5,5 ADR
 ```
 
We use the code from [https://github.com/daixiangau/acl2020-transition-discontinuous-ner](https://github.com/daixiangau/acl2020-transition-discontinuous-ner)
to format the data appropriately. This code can be found in the 
*processing_tools_from_acl2020* folder within the main directory. The tools are specific to each dataset, and you need 
to run (or create, if it is a new dataset) the corresponding *process_data.sbatch* script in the *data\raw_data\DatasetName* directory. 

The data pipeline takes in : 

1. A **text** directory containing the sentences of the dataset, one file per sentence, under the convention
*identifier.txt*
2. An **ann** directory containing the annotations corresponding to the samples bearing the relative identifiers
under the convention *identifier.ann* and the format "Tx entityType indices entityText"
3. A **split** directory containing the files *train.id*, *test.id* and *dev.id* specifying the identifiers of the samples to allocate for each sub-dataset of the split.
These files specify the identifiers of the samples to allocate for each sub-dataset of the split. For PGxCorpus, a *split.py*
file allows to randomly manage this split, with a split of 80-10-10 to change if needed

# Training 

Training is initiated by running the corresponding *run_datasetName.sbatch* script located in the main directory [PGXNER](\PGXNER), which executes the *train.py* file

List of arguments for the script : 

- Training parameters:

  - **Dataset_name** (default: PGxCorpus)
  - **Schedule** (default: linear)
  - **Number_of_epochs** (default: 30)
  - **Batch_size** (default: 8)
  - **Learning_rate** (default: 1e-05)
  - **Length_penalty** (default: 1)
  - **Warmup_ratio** (default: 0.01)
  - **Evaluation_start_epoch** (default: 100)
  - **Save_model** (Boolean) (default: 0)

- Model architecture:

  - **Target_type** ("word", "bpe", or "span," corresponding to the indices to keep in the span description after BPE encoding of the BART) (default: word)
  - **BART_name** (which follows a Huggingface model path) (default: facebook/bart-large)
  - **Decoder_type** ("avg_feature" or "avg_score") (default: avg_feature)
  - **Use_encoder_MLP** (default: 1)

- Hyperparameters:

  - **Maximum_length** (maximum output sequence length) (default: 10)
  - **Maximum_length_a** (maximum length depending on the input sequence size) (default: 1.6)
  - **Number_of_beams** (width of beam search operation) (default: 3)
  
The best model is saved in *\PGXNER\save_models*. A historic folder is created in *\PGXNER\history_dir\save models* 
containing the training history with a summary of the model, score history, and training figures.

# Inference and Evaluation 

Inference is done through the *predictor.py* script, by specifying path to the model with the best performance. The result is stored in *\PGXNER\preds* which in 
addition to predictions includes a report on the model's performance on the contiguous, nested and discontinuous parts of the dataset.

