import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from data.pipe import BartNERPipe
from model.bart import BartSeq2SeqModel
import fitlog

from fastNLP import Trainer
from model.metrics import Seq2SeqSpanMetric
from model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results
# custom fix 
import custom_utils 

from model.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
from model.generater import SequenceGeneratorModel
from fastNLP.core.sampler import  ConstTokenNumSampler
from model.callbacks import CustomFitlogCallback

fitlog.debug()
fitlog.set_log_dir('/usr/users/rattrapagemehdibenghali/benghali_meh/PGxNER/Fitlog/logs')

import argparse
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--dataset_name', default='PGxCorpus', type=str)
# Training parameters
parser.add_argument('--schedule', default='linear', type=str)
parser.add_argument('--n_epochs', default= 30 , type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--length_penalty', default=1, type=int)
parser.add_argument('--warmup_ratio', default=0.01, type=float)
parser.add_argument('--eval_start_epoch', default= 10 , type=int)
parser.add_argument('--save_model', default=0, type=int)
# Model Architecture 
parser.add_argument('--target_type', default='word', type=str)
parser.add_argument('--bart_name', default = 'facebook/bart-large', type=str)
parser.add_argument('--decoder_type', default='avg_feature', type=str)
parser.add_argument('--use_encoder_mlp', default=1, type=int)
# Hyperparameters
parser.add_argument('--max_len', default=10, type=int)
parser.add_argument('--max_len_a', default=1.6, type=float)
parser.add_argument('--num_beams', default=4, type=int)
# Logging metrics 
parser.add_argument('--history_dir', default='', type=str)

args= parser.parse_args()
print(args)
dataset_name = args.dataset_name

args.length_penalty = 1
#args.save_model = 0

""" #here
# word: ??????word???start; bpe: ???????????????bpe; span: ???????????????start end??????; span_bpe: ???????????????start?????????bpe???end?????????bpe
args.target_type = 'word'
args.bart_name = 'facebook/bart-large'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.n_epochs = 30
args.num_beams = 1
args.batch_size = 8
args.use_encoder_mlp = 1
args.lr = 1e-5
args.warmup_ratio = 0.01
eval_start_epoch = 15 """

#here
""" # the following hyper-parameters are for target_type=word
if dataset_name == 'conll2003':  # three runs get 93.18/93.18/93.36 F1
    max_len, max_len_a = 10, 0.6
elif dataset_name == 'en-ontonotes':  # three runs get 90.46/90.4/90/52 F1
    max_len, max_len_a = 10, 0.8
elif dataset_name == 'CADEC':
    max_len, max_len_a = 10, 1.6
    args.num_beams = 4
    args.lr = 2e-5
    args.n_epochs = 30
    eval_start_epoch=10
elif dataset_name == 'Share_2013':
    max_len, max_len_a = 10, 0.6
    args.use_encoder_mlp = 0
    args.num_beams = 4
    args.lr = 2e-5
    eval_start_epoch = 5
elif dataset_name == 'Share_2014':
    max_len, max_len_a = 10, 0.6
    args.num_beams = 4
    eval_start_epoch = 5
    args.n_epochs = 30
elif dataset_name == 'genia':  # three runs: 79.29/79.13/78.75
    max_len, max_len_a = 10, 0.5
    args.target_type = 'span'
    args.lr = 2e-5
    args.warmup_ratio = 0.01
elif dataset_name == 'en_ace04':  # four runs: 86.84/86.33/87/87.17
    max_len, max_len_a = 50, 1.1
    args.lr = 4e-5
elif dataset_name == 'en_ace05':  # three runs: 85.39/84.54/84.75
    max_len, max_len_a = 50, 0.7
    args.lr = 3e-5
    args.batch_size = 12
    args.num_beams = 4
    args.warmup_ratio = 0.1
elif dataset_name == 'CADEC': # The dataset of interest for this exp.
    max_len, max_len_a = 10, 1.6
    args.num_beams = 4
    args.lr = 2e-5
    args.n_epochs = 30
    eval_start_epoch=10 """


save_model = args.save_model
eval_start_epoch = args.eval_start_epoch #here
del args.save_model
lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams

length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
target_type = args.target_type
bart_name = args.bart_name
schedule = args.schedule
use_encoder_mlp = args.use_encoder_mlp

fitlog.add_hyper(args)

history_dir = args.history_dir

#######hyper
#######hyper

demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}.pt"

@cache_results(cache_fn, _refresh=True)
def get_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name, target_type=target_type)
    if dataset_name == 'conll2003':
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/dev.txt"}
        data_bundle = pipe.process_from_file(paths, demo=demo)
    elif dataset_name == 'en-ontonotes':
        paths = '../data/en-ontonotes/english'
        data_bundle = pipe.process_from_file(paths)
    else:
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

max_len, max_len_a = args.max_len, args.max_len_a #here
# print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     use_encoder_mlp=use_encoder_mlp)

vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)

import torch
# Debug Cuda 
print(f"CUDA version : {torch.version.cuda}")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=args.warmup_ratio, schedule=schedule))

if dataset_name not in ('conll2003', 'genia'):
    test_ds = data_bundle.get_dataset('test')
    for fld in test_ds.get_all_fields().keys() :
        test_ds.set_padder(fld, custom_utils.adjusted_for_np_AutoPadder()) 
    callbacks.append(CustomFitlogCallback(test_ds, raise_threshold=0.04,log_loss_every=50, history_dir = history_dir,
                                        eval_begin_epoch=eval_start_epoch, summary=args))  # If it is less than 0.04, raise exception
    eval_dataset = data_bundle.get_dataset('dev')
"""elif dataset_name == 'genia':
    dev_indices = []
    tr_indices = []
    for i in range(len(data_bundle.get_dataset('train'))):
        if i%4==0 and len(dev_indices)<1669:
            dev_indices.append(i)
        else:
            tr_indices.append(i)
    eval_dataset = data_bundle.get_dataset('train')[dev_indices]
    data_bundle.set_dataset(data_bundle.get_dataset('train')[tr_indices], name='train')
    print(data_bundle)
    callbacks.append(CustomFitlogCallback(data_bundle.get_dataset('test'), raise_threshold=0.04, history_dir = history_dir, eval_begin_epoch=eval_start_epoch, summary=args))  # ????????????0.04?????????????????????
    fitlog.add_other(name='demo', value='split dev')
else:
    callbacks.append(CustomFitlogCallback(raise_threshold=0.04, history_dir = history_dir, eval_begin_epoch=eval_start_epoch, summary=args))  # ????????????0.04?????????????????????
    eval_dataset = data_bundle.get_dataset('test')"""

sampler = None
if dataset_name in ('Share_2013',) :
    if target_type == 'bpe':
        sampler = ConstTokenNumSampler('src_seq_len', max_token=3500)
    else:
        sampler = ConstTokenNumSampler('src_seq_len', max_token=4000)
if dataset_name in ('en_ace04',) and target_type == 'bpe':
    sampler = ConstTokenNumSampler('src_seq_len', max_sentence=batch_size, max_token=2500)
elif ('large' in bart_name and dataset_name in ('en-ontonotes', 'genia')):
    sampler = ConstTokenNumSampler('src_seq_len', max_token=3000)
else:
    sampler = BucketSampler(seq_len_field_name='src_seq_len')

metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), target_type=target_type)

ds = data_bundle.get_dataset('train')
"""if dataset_name == 'conll2003':
    ds.concat(data_bundle.get_dataset('dev'))
    data_bundle.delete_dataset('dev')"""
if save_model == 1:
    save_path = 'save_models/'
else:
    save_path = None
validate_every = 100000

# Fixes
for fld in ds.get_all_fields().keys() :
    ds.set_padder(fld, custom_utils.adjusted_for_np_AutoPadder())
for fld in eval_dataset.get_all_fields().keys() :
    eval_dataset.set_padder(fld, custom_utils.adjusted_for_np_AutoPadder())

""" # Data visualisation
print(f"---------------- \n Testing dataset info \n")
ds.print_field_meta()
print(f"---------------- \n Validation dataset info \n")
eval_dataset.print_field_meta()

# Padding and batch Debug
from fastNLP import DataSetIter
from fastNLP import TorchLoaderIter
from fastNLP import BatchIter
# Debug padding 
for fld in ds.get_all_fields().keys() :
    ds.set_padder(fld, custom_utils.DebugAutoPadder())
    # ds = ds.set_ignore_type(fld, flag=False)
for fld in eval_dataset.get_all_fields().keys() :
    eval_dataset.set_padder(fld, custom_utils.DebugAutoPadder())
    eval_dataset = eval_dataset.set_ignore_type(fld, flag=False)
# Data visualisation
print(f"---------------- \n Testing dataset info \n")
ds.print_field_meta()
print(f"---------------- \n Validation dataset info \n")
eval_dataset.print_field_meta()
####
cell_0 = ds.entities.content
print(cell_0)
print(f"final dim and type : {custom_utils._get_ele_type_and_dim_debug(cell_0)}")
print(hasattr(cell_0,'dtype'))
print(hasattr(cell_0[0],'dtype'))
print(hasattr(cell_0[0][0],'dtype'))
print(type(cell_0[0][0]))
# Debug batches
batch = DataSetIter(ds, batch_size = 2)
print("-------------------------------- \n Printing out the content of data \n")
print(ds['entities'][0])
print(type(ds))
print("-------------------------------- \n Printing out the content of batch \n")
print(batch.dataset[0])
print(type(batch))
num_batch = len(batch)
print(f"{num_batch} batches") 
for batch_x,batch_y in batch:
    print(batch_x) """

trainer = Trainer(train_data=ds, model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=4, n_epochs=n_epochs, print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 100,
                  dev_data=eval_dataset, metrics=metric, metric_key='f',
                  validate_every=validate_every, save_path=save_path, use_tqdm='SEARCH_OUTPUT_FP' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm='SEARCH_OUTPUT_FP' not in os.environ,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size*2)

trainer.train(load_best_model=False)

