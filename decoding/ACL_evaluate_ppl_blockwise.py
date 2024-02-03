import os
import copy
import numpy as np
import json
import argparse
import random
import scipy
import config
from LLAMA import LLAMA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import utils_llama.activation as ana
import scipy
import math
import time
import pickle
import datasets
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)

model_dir = '/ossfs/workspace/nas/gzhch/data/models/Llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    device_map='auto',
    torch_dtype=torch.float16,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

## load cached llm act if possible
log_dir = '/ossfs/workspace/cache_v2'
cache_dir = '/ossfs/workspace/nas/gzhch/data/cache'
llama = LLAMA(model, tokenizer, cache_dir)

def load_data(task_name, n_shot=1, seed=42):
    data_dirs = {
        'xsum' : '/ossfs/workspace/nas/gzhch/data/datasets/xsum',
        'gsm8k' : '/ossfs/workspace/nas/gzhch/data/datasets/gsm8k',
        'alpaca' : '/ossfs/workspace/nas/gzhch/data/datasets/alpaca',
        'wmt' : '/ossfs/workspace/nas/gzhch/data/datasets/wmt14_de-en_test',
        'wikitext2' : '/ossfs/workspace/nas/gzhch/data/datasets/wikitext-2-v1',
        'wikitext_dense' : '/ossfs/workspace/nas/gzhch/data/datasets/wikitext-2-v1',
        'wikitext_eval' : '/ossfs/workspace/nas/gzhch/data/datasets/wikitext-2-v1',
    }
    if task_name == 'gsm8k':
        dataset = datasets.load_dataset(data_dirs[task_name])
    elif task_name == 'wikitext2':
        dataset = datasets.load_from_disk(data_dirs[task_name])
        dataset = dataset['train'].filter(lambda x: len(x['text'])>100) 
        dataset = dataset.select(random.sample(range(len(dataset)), 1000))

    elif task_name == 'wikitext_eval':
        dataset = datasets.load_from_disk(data_dirs[task_name])
        dataset = dataset['test'].filter(lambda x: len(x['text'])>100) 

    elif task_name == 'wikitext_dense':
        def tokenize_texts(examples):
            tokenized_inputs = tokenizer(examples["text"])
            return tokenized_inputs

        def group_texts(examples):
            # Concatenate all texts.
            max_length = 1024
            concatenated_examples = {k: list(chain(*examples[k])) for k in ['input_ids']}
            total_length = len(concatenated_examples['input_ids'])
            # print(len(concatenated_examples['input_ids']), '\n')
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_length:
                total_length = (total_length // max_length) * max_length
            # else:
                # print('aaa')
            # Split by chunks of max_len.
            # result = {
            #     k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            #     for k, t in concatenated_examples.items()
            # }
            result = {'input_ids': [concatenated_examples['input_ids'][i : i + max_length] for i in range(0, total_length, max_length)]}
            return result

        dataset = datasets.load_from_disk(data_dirs[task_name])
        dataset = dataset.map(tokenize_texts, batched=True, num_proc=4)
        dataset = dataset.map(group_texts, batched=True, num_proc=4, remove_columns=['text', 'attention_mask'])
        dataset['train'] = dataset['train'].shuffle(seed=seed)

    return dataset

# 创建一个简单的两层全连接神经网络
class Projector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.SiLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        return out

class LinearProjector(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearProjector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        # self.act = nn.SiLU()
    
    def forward(self, x):
        out = self.fc(x)
        # out = self.act(out)
        return out


@torch.no_grad()
def eval(x, y, net):
    output = net(x)
    loss = criterion(output, y)
    return loss

def deduplication(data):
    tokens = data['context'][:, 5]
    unique_tokens = []
    unique_token_ids = []
    for idx in range(len(tokens)):
        if tokens[idx] not in unique_tokens:
            unique_tokens.append(tokens[idx])
            unique_token_ids.append(idx)
    random.shuffle(unique_token_ids)
    ids = unique_token_ids

    return {k : v[ids] for k, v in data.items()}

def train(net, train_set, stim_neurons=None, resp_neurons=None, max_step=100000):
    logs = []
    # layer1, layer2 = 10, 15
    total_batch = len(train_set) // batch_size

    total_batch = min(total_batch, max_step)
    for b in range(total_batch):
        input_ids = train_set[b * batch_size: (b + 1) * batch_size]['input_ids']
        input_ids = torch.tensor(input_ids)
        input = dict(input_ids=input_ids, attention_mask=torch.ones(input_ids.shape))
        with torch.no_grad():
            res = llama.get_neuron_activation_and_loss(input)

        if stim_neurons is not None:
            X = res['ffn_gate'][:, layer1, stim_neurons].cuda().float()
        else:
            X = res['ffn_gate'][:, layer1, :].cuda().float()
        if resp_neurons is not None:
            Y = res['ffn_gate'][:, layer2, resp_neurons].cuda().float()
        else:
            Y = res['ffn_gate'][:, layer2, :].cuda().float()

        output = net(X)
        loss = criterion(output, Y)
        
        optimizer.zero_grad() 
        (loss * output.shape[1]).backward()        
        optimizer.step()       
        
        if (b+1) % 1 == 0:
            eval_loss = eval(test_X.cuda(), test_Y.cuda(), net).item()
            print(f'Epoch [{b+1}/{total_batch}], Train Loss: {loss.item():.6f}, Eval Loss: {eval_loss:.6f}')
            logs.append(f'Epoch [{b+1}/{total_batch}], Train Loss: {loss.item():.6f}, Eval Loss: {eval_loss:.6f}')
    return logs

def evaluate_ppl(eval_data, model, fake_ffn=None, num_of_batch=3, **forwrd_args):
    ppls = []
    batch_size = 100
    for b in range(num_of_batch):
        input = tokenizer(eval_data['text'][b * batch_size: (b + 1) * batch_size], padding='longest', return_tensors='pt')
        result = ana.custom_forward(model, input['input_ids'].cuda(), inspect_acts=['ffn_gate'], fake_ffn=fake_ffn, **forwrd_args)
        logits = result['logits']
        labels = input['input_ids']
        input_ids = input['input_ids'][:, :-1]

        # calculate loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, 32000)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        loss = loss_fct(shift_logits, shift_labels).view(labels.shape[0], -1)
        t = (loss * input['attention_mask'][:, :-1]).sum(dim=1)/input['attention_mask'].sum(dim=1)
        ppls += torch.exp(t).tolist()
    ppl = torch.nan_to_num(torch.tensor(ppls)).mean().tolist()
    return ppl


eval_data = load_data('wikitext_eval')

log_file = open('/ossfs/workspace/nas/gzhch/br/ACL_result/blockwise_proj_direct_noskip_ppl.jsonl', 'a')
# log_file = open('/ossfs/workspace/nas/gzhch/br/ACL_result/blockwise_proj_noskip_ppl.jsonl', 'a')

random_net = LinearProjector(11008, 11008).half()
zero_net = LinearProjector(11008, 11008).half()
zero_net.fc.weight.data = torch.zeros(zero_net.fc.weight.data.shape).half()
zero_net.fc.bias.data = torch.zeros(zero_net.fc.bias.data.shape).half()


block_size = 8

for layer1 in range(0, 32, block_size):
    projs = []
    random_projs = []
    zero_projs = []
    # for layer2 in range(layer1 + 2, layer1 + block_size - 1, 2):
    for layer2 in range(layer1 + 2, layer1 + block_size - 1, 1):
        layer1_tmp = layer2 - 1
        proj = ana.FFNProjector(layer1_tmp, layer2, torch.load(f'/ossfs/workspace/cache_v2/net_{layer1_tmp}_{layer2}.pt'))
        random_proj = ana.FFNProjector(layer1_tmp, layer2, random_net)
        zero_proj = ana.FFNProjector(layer1_tmp, layer2, zero_net)
        projs.append(proj)
        random_projs.append(random_proj)
        zero_projs.append(zero_proj)
    ppl = evaluate_ppl(eval_data, model, projs, fake_ffn_direct_contribution_only=True)
    ppl_random = evaluate_ppl(eval_data, model, random_projs, fake_ffn_direct_contribution_only=True)
    ppl_zero = evaluate_ppl(eval_data, model, zero_projs, fake_ffn_direct_contribution_only=True)

    # ppl = evaluate_ppl(eval_data, model, projs)
    # ppl_random = evaluate_ppl(eval_data, model, random_projs)
    # ppl_zero = evaluate_ppl(eval_data, model, zero_projs)

    json.dump(dict(layer1=layer1, ppl=ppl, ppl_random=ppl_random, ppl_zero=ppl_zero), log_file)
    log_file.write('\n')
    log_file.flush()
        

