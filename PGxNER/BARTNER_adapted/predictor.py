import torch
from fastNLP import cache_results
from data.pipe import BartNERPipe
from fastNLP import SequentialSampler, SortedSampler

from fastNLP import DataSetIter
from fastNLP.core.utils import _move_dict_value_to_device
from tqdm import tqdm
import json

# custom fix 
import custom_utils 

dataset_name = 'PGxCorpus'
model_path = 'save_models/best_SequenceGeneratorModel_f_2023-02-19-17-46-23-365303'  # you can set args.save_model=1 in train.py
bart_name = 'facebook/bart-large'
target_type = 'word'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}.pt"


@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name, target_type=target_type)
    if dataset_name == 'conll2003':
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/dev.txt"}
        data_bundle = pipe.process_from_file(paths, demo=False)
    elif dataset_name == 'en-ontonotes':
        paths = '../data/en-ontonotes/english'
        data_bundle = pipe.process_from_file(paths)
    else:
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=False)
    return data_bundle, pipe.tokenizer, pipe.mapping2id


data_bundle, tokenizer, mapping2id = get_data()

model = torch.load(model_path)

device = torch.device(device)
model.to(device)
model.eval()

eos_token_id = 0
word_start_index = len(mapping2id) + 2
not_bpe_start = 0

"""if dataset_name == 'PGxCorpus':  # if you use other dataset, please change this mapping
    mapping = {
        '<<location>>': 'LOC',
        '<<person>>': 'PER',
        '<<organization>>': 'ORG',
        '<<others>>': 'MISC'
    }"""

id2label = {k: v for k, v in enumerate(mapping2id.keys())} # Change here


def get_pairs(ps, word_start_index, target_type):
    pairs = []
    cur_pair = []
    for j in ps:
        if j < word_start_index:
            if target_type == 'span':
                if len(cur_pair) > 0 and len(cur_pair) % 2 == 0:
                    if all([cur_pair[i] <= cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                        pairs.append(tuple(cur_pair + [j]))
            else:
                if len(cur_pair) > 0:
                    if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                        pairs.append(tuple(cur_pair + [j]))
            cur_pair = []
        else:
            cur_pair.append(j)
    return pairs


def get_spans(pairs, cum_lens, mapping2id, dataset_name, id2label):
    spans = []
    pred_y = ['O' for _ in range(len(raw_words_i))]
    for pair in pairs:
        label = pair[-1]
        try:
            idxes = [cum_lens.index(p - len(mapping2id) - 2) for p in pair[:-1]]
            start_idx = idxes[0]
            end_idx = idxes[-1]
            if dataset_name in ('en_ace04', 'en_ace05'):
                spans.append((start_idx, end_idx, id2label[label - 2]))
            elif dataset_name in ('PGxCorpus') :
                """if torch.is_tensor(start_idx) :
                    start_idx = start_idx.item()
                    print(f"start idx : {start_idx}")
                if torch.is_tensor(end_idx) :
                    end_idx = end_idx.item()
                    print(f"end idx : {end_idx}")"""
                if torch.is_tensor(label) :
                    label = label.item()
                spans.append((idxes, id2label[label - 2]))
            else:
                pred_y[start_idx] = f'B-{id2label[label - 2]}'
                for _ in range(start_idx + 1, end_idx + 1):
                    pred_y[_] = f'I-{id2label[label - 2]}'
        except Exception as e:
            pass
    return spans,pred_y

for name in ['test']:
    ds = data_bundle.get_dataset(name)
    for fld in ds.get_all_fields().keys() :
        ds.set_padder(fld, custom_utils.adjusted_for_np_AutoPadder())
    print(ds)
    #ds.set_ignore_type('raw_words', 'raw_target')
    #ds.set_target('raw_words', 'raw_target')
    ds.set_ignore_type('raw_words')
    ds.set_target('raw_words')
    with open(f'preds/{name}.PGxCorpus', 'w', encoding='utf-8') as f:
        data_iterator = DataSetIter(ds, batch_size=32, sampler=SequentialSampler())
        discontinuous = {'r':0, 'count':0, 'correct_preds':0, 'incorrect_preds':0}
        contiguous =  {'r':0, 'count':0, 'correct_preds':0, 'incorrect_preds':0}
        assumptions = {'assumed_disc (candidate)' :0,'assumed_flat (candidate)' :0, 'assumed_flat (default)':0}
        for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            first = batch_x['first']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            raw_words = batch_y['raw_words']
            #raw_targets = batch_y['raw_target']
            pred_y = model.predict(src_tokens=src_tokens, src_seq_len=src_seq_len, first=first)
            pred = pred_y['pred']
            tgt_tokens = batch_y['tgt_tokens']
            pred_eos_index = pred.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
            pred = pred[:, 1:]  # 去掉</s>
            tgt_tokens = tgt_tokens[:, 1:]
            pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
            pred_seq_len = (pred_seq_len - 2).tolist()
            tgt_seq_len = (tgt_seq_len - 2).tolist()
            for i, ps in enumerate(pred.tolist()):
                em = 0
                ps = ps[:pred_seq_len[i]]
                ts = tgt_tokens[i, :tgt_seq_len[i]]
                pairs, t_pairs = [], []
                if len(ps):
                    pairs = get_pairs(ps, word_start_index, target_type)
                if len(ts):
                    t_pairs = get_pairs(ts, word_start_index, target_type)
                raw_words_i = raw_words[i]
                src_tokens_i = src_tokens[i, :src_seq_len[i]].tolist()
                src_tokens_i = tokenizer.convert_ids_to_tokens(src_tokens_i)
                cum_lens = [1]
                start_idx = 1
                for token in raw_words_i:
                    start_idx += len(tokenizer.tokenize(token, add_prefix_space=True))
                    cum_lens.append(start_idx)
                cum_lens.append(start_idx + 1)

                #target_y = raw_targets[i]
                pred_spans, pred_y = get_spans(pairs, cum_lens, mapping2id, dataset_name, id2label)
                target_spans, _ = get_spans(t_pairs, cum_lens, mapping2id, dataset_name, id2label)
                pred_entities = [[raw_words_i[idx] for idx in span[0]]+[span[1]] for span in pred_spans]
                target_entities = [[raw_words_i[idx] for idx in span[0]]+[span[1]] for span in target_spans]
                if dataset_name in ('en_ace04', 'en_ace05', "PGxCorpus"):
                    f.write(f"""'sentence': {' '.join(raw_words_i)},
                    'spans': {target_spans},
                    'pred_spans': {pred_spans},
                    'entities' : {[' '.join(ent) for ent in target_entities]},
                    'pred_entities' : {[' '.join(ent) for ent in pred_entities]} \n""")
                else:
                    #assert len(pred_y) == len(raw_words_i) == len(target_y)
                    #for raw_word, t, p in zip(raw_words_i, target_y, pred_y):
                        #f.write(f'{raw_word} {t} {p}\n')
                    assert len(pred_y) == len(raw_words_i)
                    for raw_word, p in zip(raw_words_i, pred_y):
                        f.write(f'{raw_word} {p}\n')

                # Per NER category metric computation        
                if dataset_name in ("PGxCorpus"):
                    # Recall 
                    for t_span in target_spans: 
                        if sorted(t_span[0]) == list(range(min(t_span[0]), max(t_span[0])+1)):
                            contiguous['count']+=1
                            if t_span in pred_spans : 
                                contiguous['r']+=1
                        else :
                            discontinuous['count']+=1
                            if t_span in pred_spans : 
                                discontinuous['r']+=1
                    # Precision
                    for p_span in pred_spans : 
                        # Correct prediction cases
                        if p_span in target_spans :
                            if sorted(p_span[0]) == list(range(min(p_span[0]), max(p_span[0])+1)):
                                contiguous['correct_preds']+=1
                            else : 
                                discontinuous['correct_preds']+=1
                        # Incorrect prediction cases 
                        else : 
                            candidate = None
                            for t_span in target_spans :
                                if t_span[0][0] == p_span[0][0] : # Assume span with same start idx as pred is the correct one
                                    candidate = t_span 
                            if candidate is not None: 
                                if sorted(candidate[0])==list(range(min(candidate[0]),max(candidate[0])+1)): 
                                    contiguous['incorrect_preds']+=1
                                    assumptions['assumed_flat (candidate)']+=1
                                else :
                                    discontinuous['incorrect_preds']+=1
                                    assumptions['assumed_disc (candidate)']+=1
                            else : 
                                contiguous['incorrect_preds']+=1
                                assumptions['assumed_flat (default)']+=1                                                                                                
                f.write('\n')
        f.write(f"True contiguous count : {contiguous['count']}. Contiguous recall : {contiguous['r']/contiguous['count']}\n")
        f.write(f"True discontinuous count : {discontinuous['count']}. Discontinuous recall : {discontinuous['r']/discontinuous['count']}\n")
        f.write(f"Estimated contiguous precision : {contiguous['correct_preds']/(contiguous['incorrect_preds']+contiguous['correct_preds'])}\n")
        f.write(f"Estimated discontinuous precision : {discontinuous['correct_preds']/(discontinuous['incorrect_preds']+discontinuous['correct_preds'])}\n")
        f.write(f"Estimated {discontinuous['incorrect_preds']+discontinuous['correct_preds']} predictions on discontinuous entities, \n\t{assumptions['assumed_disc (candidate)']} of them were approximated from candidate spans\n")
        f.write(f"""Estimated {contiguous['incorrect_preds']+contiguous['correct_preds']} predictions on continuous entities, 
{assumptions['assumed_flat (candidate)']} of them were approximated from candidate spans,
{assumptions['assumed_flat (default)']} were considered discontinuous by default""")

print(f"In total, has {not_bpe_start} predictions on the non-word start.")

# the output file for flat NER will be similar to the following(empty line separate two sentences)
# SOCCER O O
# - O O
# JAPAN B-LOC B-LOC
# GET O O
# LUCKY O O
# WIN O O
# , O O
# CHINA B-PER B-LOC


# the output file for nested NER will be similar to the following(empty line separate two sentences)
#  {'sentence': "xxx xxx", 'pred_spans': [(start, end, label), (start, end, label)], 'spans': [(start, end, label)...]}
#  {'sentence': "xxx xxx", 'pred_spans': [(start, end, label), (start, end, label)], 'spans': [(start, end, label)...]}
