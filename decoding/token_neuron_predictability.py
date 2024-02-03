import os
import copy
import numpy as np
import json
import argparse
import random
import scipy
import config
from GPT import GPT
from LLAMA import LLAMA
from StimulusModel import LMFeatures
from utils_stim import get_story_wordseqs
# from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge, ridge_corr
from utils_ridge.ridge_torch import ridge_torch, bootstrap_ridge_torch, ridge_corr_torch
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D, lanczosinterp2D_torch
from utils_ridge.util import make_delayed
from utils_ridge.utils import mult_diag, counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import utils_llama.activation as ana

import scipy
import math
import matplotlib.pyplot as plt

import time
import h5py

random.seed(42)

class ARGS:
    def __init__(self):
        self.subject = 'S1'
        self.sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
        self.layer = 20
        self.act_name = 'ffn_gate'

args = ARGS()

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type = str, required = False)
parser.add_argument("--sessions", nargs = "+", type = int, 
    default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
parser.add_argument("--act_name", type = str, default = "ffn_gate")
parser.add_argument("--window", type = int, required = True)
parser.add_argument("--layer", type = int)
parser.add_argument("--runs", type = int, default = 3)
parser.add_argument("--chunk", type = int, default = 2)

args = parser.parse_args()

# torch.cuda.memory._record_memory_history()
torch.cuda.empty_cache()
torch.set_grad_enabled(False)

# training stories
stories = []
with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    sess_to_story = json.load(f) 
for sess in args.sessions:
    stories.extend(sess_to_story[str(sess)])

stories = stories[:10]

all_words = []
word_seqs = get_story_wordseqs(stories)
for story in stories:
    all_words += word_seqs[story].data



model_dir = '/ossfs/workspace/nas/gzhch/data/models/Llama-2-7b-hf'
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir, 
#     device_map='auto',
#     torch_dtype=torch.float16,
# ).eval()

model = None

tokenizer = AutoTokenizer.from_pretrained(model_dir)


## load cached llm act if possible
cache_dir = '/ossfs/workspace/act_cache'
llama = LLAMA(model, tokenizer, cache_dir)


log_name = f'llama_2_7b_token_neuron_predict-{args.window}.jsonl'
log_dir = os.path.join(config.RESULT_DIR, log_name)
log_file = open(log_dir, 'w')

def get_clm_loss(args, stories, llama):
    clm_loss = {}
    word_seqs = get_story_wordseqs(stories)
    for story in stories:
        words = word_seqs[story].data
        clm_loss[story] = llama.get_clm_loss(story, words, args.window, chunk=args.chunk)
    clm_loss_aggr = torch.vstack([torch.tensor(clm_loss[story]) for story in stories]).float().cuda()
    return clm_loss_aggr

def get_stim_torch(args, stories, llama):
    word_seqs = get_story_wordseqs(stories)
    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        embs = llama.get_llm_act(story, words, args.window, args.act_name, args.layer, chunk=args.chunk, cache_all_layer=False)
        word_vecs[story] = embs
    ds_mat = torch.vstack([torch.tensor(word_vecs[story]) for story in stories]).float().cuda()
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    
    return ds_mat, r_mean, r_std

def normalize(ds_mat, r_mean, r_std):
    return torch.nan_to_num(torch.matmul((ds_mat - r_mean), torch.linalg.inv(torch.diag(r_std))))

def layerwise_alignment(stim, resp, clm_loss_aggr, alphas='adaptive', runs=1, n_train=1000, n_test=500, words=all_words):
    r_mean, r_std = resp.mean(0), resp.std(0)
    r_std[r_std == 0] = 1

    result = {}
    

    for run in range(runs):
        n_total = stim.shape[0]

        chosed = []
        chosed_ids = []
        while len(chosed_ids) < n_train + n_test:
            idx = random.choice(range(len(words)))
            if all_words[idx] not in chosed:
                chosed.append(all_words[idx])
                chosed_ids.append(idx)
        ids = chosed_ids

        tstim, hstim = stim[ids[:n_train]], stim[ids[n_train:]]
        tresp, hresp = resp[ids[:n_train]], resp[ids[n_train:]]

        if alphas is None:
            alphas = torch.tensor([1 for _ in range(resp.shape[-1])]).cuda()

        elif alphas == 'adaptive':
            nchunks = int(np.ceil(tresp.shape[0] / 5 / 100))
            weights, alphas, bscorrs = bootstrap_ridge_torch(tstim, tresp, use_corr = False, alphas = np.logspace(0, 3, 10),
                    nboots = 3, chunklen = 100, nchunks = nchunks)        

        bs_weights = ridge_torch(tstim, tresp, alphas)
        bs_weights = bs_weights.to(hstim.device).to(hstim.dtype)
        pred = hstim.matmul(bs_weights)

        ### evaluate
        pred = pred.cpu()
        hresp = hresp.cpu()

        test_loss = clm_loss_aggr[ids[n_train:], :].cpu()
        word_pearson, word_p = [], []
        for i in range(pred.shape[0]):
            stat = scipy.stats.pearsonr(pred[i, :].flatten(), hresp[i, :].flatten())
            word_pearson.append(stat.statistic)
            word_p.append(stat.pvalue)
        word_pearson = torch.tensor(word_pearson)
        word_std = (pred-hresp).std(dim=1)

        for pos in [1, 2, 3, 4, 5]:
            for k in [10, 20, 30, 50, 100, 200, 500]:
                sorted_index = test_loss[:, -pos].topk(n_test).indices
                current_pearson = word_pearson[sorted_index]
                current_std = word_std[sorted_index]

                key = f'top_{k}-pos_{pos}-pearson'
                if key not in result.keys():
                    result[f'top_{k}-pos_{pos}-std'] = []
                    result[f'top_{k}-pos_{pos}-pearson'] = []
                    result[f'bottom_{k}-pos_{pos}-std'] = []
                    result[f'bottom_{k}-pos_{pos}-pearson'] = []

                result[f'top_{k}-pos_{pos}-pearson'].append((current_pearson[:k].mean().item(), current_pearson[:k].std().item()))
                result[f'top_{k}-pos_{pos}-std'].append((current_std[:k].mean().item(), current_std[:k].std().item()))
                result[f'bottom_{k}-pos_{pos}-pearson'].append((current_pearson[-k:].mean().item(), current_pearson[-k:].std().item()))
                result[f'bottom_{k}-pos_{pos}-std'].append((current_std[-k:].mean().item(), current_std[-k:].std().item()))

    return result

def calculate_diff(x, y):
    std = (x-y).std().item()
    weighted_std = ((x-y)*y).std().item()
    pearson = scipy.stats.pearsonr(x.reshape(-1).cpu(), y.reshape(-1).cpu())
    return dict(std=std, weighted_std=weighted_std, pearson=pearson.statistic, p = pearson.pvalue)


args2 = copy.deepcopy(args)


clm_loss_aggr = get_clm_loss(args, stories, llama)

for layer1 in range(0, 31, 2):
# for layer1 in [10]:
    args.layer = layer1
    rstim, r_mean, r_std = get_stim_torch(args, stories, llama)

    for layer2 in range(0, 31, 2):
    # for layer2 in [16]:
        args2.layer = layer2
        rresp, r_mean, r_std = get_stim_torch(args2, stories, llama)

        res = layerwise_alignment(rstim, rresp, clm_loss_aggr, alphas='adaptive', runs=args.runs, n_train=1000, n_test=1000)
        res.update(dict(layer1=layer1, layer2=layer2, window=args.window))
    
        json.dump(res, log_file)
        log_file.write('\n')
        log_file.flush()

        # print(layer1, layer2, res['all-pearson-mean'], res_norm['all-pearson-mean'])