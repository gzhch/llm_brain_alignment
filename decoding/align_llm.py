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
from utils_stim import get_stim, get_story_wordseqs
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

class ARGS:
    def __init__(self):
        self.subject = 'S1'
        self.gpt = 'perceived'
        self.sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
        self.layer = 20
        self.act_name = 'ffn_gate'

args = ARGS()

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type = str, required = True)
parser.add_argument("--gpt", type = str, default = "perceived")
parser.add_argument("--sessions", nargs = "+", type = int, 
    default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
parser.add_argument("--layer", type = int, required = True)
parser.add_argument("--act_name", type = str, required = True)
parser.add_argument("--window", type = int, required = True)
parser.add_argument("--layer2", type = int, required = True)
parser.add_argument("--chunk", type = int, default = 0)



args = parser.parse_args()

log_name = f'innerllm_{args.window}-layer_{args.layer}_{args.layer2}-{args.act_name}.txt'
log_dir = os.path.join(config.RESULT_DIR, log_name)
log_file = open(log_dir, 'w')

# def get_stim(args, stories, llama, tr_stats = None, delay=True, vox=None):
#     word_seqs = get_story_wordseqs(stories)
#     word_vecs = {}
#     for story in stories:
#         words = word_seqs[story].data
#         embs = llama.get_llm_act(story, words, args.window, args.act_name, args.layer)
#         word_vecs[story] = embs
    
#     word_mat = torch.vstack([torch.tensor(word_vecs[story]) for story in stories]).cuda()
#     # word_mat = np.vstack([word_vecs[story] for story in stories])
#     word_mean, word_std = word_mat.mean(0), word_mat.std(0)

#     ds_vecs = {story : lanczosinterp2D(word_vecs[story], word_seqs[story].data_times, word_seqs[story].tr_times) 
#                for story in stories}
#     ds_mat = np.vstack([ds_vecs[story][5+config.TRIM:-config.TRIM] for story in stories])

#     if vox is not None:
#         ds_mat = ds_mat[:, vox]

#     if tr_stats is None: 
#         r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
#         r_std[r_std == 0] = 1
#     else: 
#         r_mean, r_std = tr_stats
#     ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
#     if delay:
#         del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
#     else:
#         del_mat = ds_mat
#     if tr_stats is None: return del_mat, (r_mean, r_std), (word_mean, word_std)
#     else: return del_mat, None, None

def get_stim_torch(args, stories, llama, tr_stats = None, delay=True, vox=None):
    word_seqs = get_story_wordseqs(stories)
    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        embs = llama.get_llm_act(story, words, args.window, args.act_name, args.layer, chunk=args.chunk)
        word_vecs[story] = embs
    word_mat = torch.vstack([torch.tensor(word_vecs[story]) for story in stories]).float()
    word_mean, word_std = word_mat.mean(0), word_mat.std(0)

    ds_vecs = {story : lanczosinterp2D_torch(word_vecs[story], word_seqs[story].data_times, word_seqs[story].tr_times) 
                for story in stories}
    ds_mat = torch.vstack([ds_vecs[story][5+config.TRIM:-config.TRIM] for story in stories])


    if vox is not None:
            ds_mat = ds_mat[:, vox]

    if tr_stats is None: 
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else: 
        r_mean, r_std = tr_stats
    ds_mat = torch.nan_to_num(torch.matmul((ds_mat - r_mean), torch.linalg.inv(torch.diag(r_std))))
    if delay:
        del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    else:
        del_mat = ds_mat
    if tr_stats is None: return del_mat, (r_mean, r_std), (word_mean, word_std)
    else: return del_mat, None, None

# training stories
stories = []
with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    sess_to_story = json.load(f) 
for sess in args.sessions:
    stories.extend(sess_to_story[str(sess)])

stories = stories[:40]

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


args2 = copy.deepcopy(args)
args2.layer = args.layer2

rstim, tr_stats, word_stats = get_stim_torch(args, stories, llama, delay=False)
rresp, tr_stats, word_stats = get_stim_torch(args2, stories, llama, delay=False)

rstim = rstim.cuda()
rresp = rresp.cuda()

nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))

weights, alphas, bscorrs = bootstrap_ridge_torch(rstim, rresp, use_corr = False, alphas = np.logspace(0, 2, 10),
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
bscorrs = bscorrs.mean(2).max(0)
vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])

del rstim, rresp

stim_dict = {story : get_stim_torch(args, [story], llama, delay=False)[0] for story in stories}
resp_dict = {story : get_stim_torch(args2, [story], llama, delay=False)[0] for story in stories}

# noise_model = torch.zeros([len(vox), len(vox)]).cuda()
for hstory in stories[:10]:
    tstim, hstim = torch.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
    tresp, hresp = torch.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
    tstim, hstim = tstim.cuda(), hstim.cuda()
    tresp, hresp = tresp.cuda(), hresp.cuda()
    bs_weights = ridge_torch(tstim, tresp, alphas)
    bs_weights = bs_weights.to(hstim.device).to(hstim.dtype)
    pred = hstim.matmul(bs_weights)
    # resids = hresp - pred
    # bs_noise_model = resids.T.matmul(resids)
    # noise_model += bs_noise_model / torch.diag(bs_noise_model).mean() / len(stories)

    pred = pred.cpu().numpy() 
    hresp = hresp.cpu().numpy()
    log_file.write(str(scipy.stats.pearsonr(pred.flatten(), hresp.flatten())))
    log_file.write('\n')
    print(scipy.stats.pearsonr(pred.flatten(), hresp.flatten()))
