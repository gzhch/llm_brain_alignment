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
parser.add_argument("--layer2", type = int, required = True)
args = parser.parse_args()

torch.manual_seed(0)

model_dir = '/ossfs/workspace/nas/gzhch/data/models/Llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    device_map='auto',
    torch_dtype=torch.float16,
).eval()

# model = None

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
    }
    if task_name == 'gsm8k':
        dataset = datasets.load_dataset(data_dirs[task_name])
    elif task_name == 'wikitext2':
        dataset = datasets.load_from_disk(data_dirs[task_name])
        dataset = dataset['train'].filter(lambda x: len(x['text'])>100) 
        dataset = dataset.select(random.sample(range(len(dataset)), 1000))

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


# load dataset
wiki_data = load_data('wikitext_dense')


 
train_set = wiki_data['train']
stim_neurons = None
resp_neurons = None
input_size = 11008 if stim_neurons is None else len(stim_neuron)
output_size = 11008 if resp_neurons is None else len(resp_neuron)
batch_size = 10
max_step = 100
lr = 0.01



# get test data once and for all
test_data = []
for b in range(5):
    input_ids = wiki_data['validation'][b * batch_size: (b + 1) * batch_size]['input_ids']
    input_ids = torch.tensor(input_ids)
    input = dict(input_ids=input_ids, attention_mask=torch.ones(input_ids.shape))
    with torch.no_grad():
        res = llama.get_neuron_activation_and_loss(input)
        test_data.append(res)
test_data = {k: torch.cat([i[k] for i in test_data]) for k in test_data[0].keys()}


# layer2 = args.layer2
projectors = []
# for layer1 in range(0, 32, 2):
# for layer1 in range(1, 31, 2):
#     layer2 = layer1 + 2
for layer1 in range(0, 31, 1):
    layer2 = layer1 + 1
    net = LinearProjector(input_size, output_size).cuda()
    proj = dict(layer1=layer1, 
                layer2=layer2,
                net=net,
                optimizer=optim.Adagrad(net.parameters(), lr=lr),
                log=open(os.path.join(log_dir, f'{layer1}-{layer2}.txt'), 'w'))
    projectors.append(proj)

criterion = nn.MSELoss()

### get text set
test_X, test_Y = [], []


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

    for proj in projectors:
        # print('xxxx')
        layer1 = proj['layer1']
        layer2 = proj['layer2']
        # net = proj['net']
        # optimizer = proj['optimizer']

        if stim_neurons is not None:
            X = res['ffn_gate'][:, layer1, stim_neurons].cuda().float()
            test_X = test_data['ffn_gate'][:, layer1, stim_neurons].cuda().float()
        else:
            X = res['ffn_gate'][:, layer1, :].cuda().float()
            test_X = test_data['ffn_gate'][:, layer1, :].cuda().float()
        if resp_neurons is not None:
            Y = res['ffn_gate'][:, layer2, resp_neurons].cuda().float()
            test_Y = test_data['ffn_gate'][:, layer2, resp_neurons].cuda().float()
        else:
            Y = res['ffn_gate'][:, layer2, :].cuda().float()
            test_Y = test_data['ffn_gate'][:, layer2, :].cuda().float()

        output = proj['net'](X)
        loss = criterion(output, Y)
    
        proj['optimizer'].zero_grad() 
        (loss * output.shape[1]).backward()        
        proj['optimizer'].step()       
        
        if (b+1) % 1 == 0:
            eval_loss = eval(test_X.cuda(), test_Y.cuda(), proj['net']).item()
            log_string = f'{layer1} {layer2} Step [{b+1}/{total_batch}], Train Loss: {loss.item():.6f}, Eval Loss: {eval_loss:.6f}\n'
            print(log_string)
            proj['log'].write(log_string)
            proj['log'].flush()

for proj in projectors:
    layer1 = proj['layer1']
    layer2 = proj['layer2']
    save_path = os.path.join(log_dir, f'net_{layer1}_{layer2}.pt')
    torch.save(proj['net'].half(), save_path)

