import os
import torch
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.nn.functional import softmax
import utils_llama.activation as ana

class LLAMA():    
    def __init__(self, model, tokenizer, cache_dir): 
        self.model = model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        # self.vocab = vocab
        # self.word2id = {w : i for i, w in enumerate(self.vocab)}
        # self.UNK_ID = self.word2id['<unk>']

    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        
    def get_story_array(self, words, context_size, context_token=True):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_size + 1
        enc = self.tokenizer(words, is_split_into_words=True)
        story_ids = enc['input_ids']
        story_array = np.zeros([len(words), nctx]) #+ self.UNK_ID
        if context_token:
            for i in range(len(story_array)):
                token_span = enc.word_to_tokens(i)
                if token_span is None:
                    story_array[i] = story_array[i - 1]
                else:
                    segment = story_ids[max(token_span[1]-nctx, 0) : token_span[1]]
                    # segment = story_ids[i:i+nctx]
                    story_array[i, -len(segment):] = segment
        else:
            raise NotImplementError
        return torch.tensor(story_array).long()

    def get_neuron_activation_and_loss(self, input):
        model = self.model
        result = ana.custom_forward(model, input['input_ids'].cuda(), inspect_acts=['ffn_gate'])
        logits = result['logits']
        labels = input['input_ids']
        input_ids = input['input_ids'][:, :-1]

        # calculate loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, 32000)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        loss = loss_fct(shift_logits, shift_labels).view(labels.shape[0], -1)

        b = 5
        mask = input['attention_mask'][:, :-1] == 1
        loss = loss * mask + -100 * (~mask)
        input_ids = input_ids * mask + -100 * (~mask)
        expanded_loss = torch.cat([torch.ones(loss.shape[0], b) * -100, loss, torch.ones(loss.shape[0], b) * -100], dim=1)
        expanded_input_ids = torch.cat([torch.ones(input_ids.shape[0], b) * -100, input_ids, torch.ones(input_ids.shape[0], b) * -100], dim=1).int()

        # signal delay
        losses = []
        context = []
        for offset in range(2 * b):
            losses.append(expanded_loss[:, offset: offset + loss.shape[1]])
            context.append(expanded_input_ids[:, offset: offset + loss.shape[1]])
        losses = torch.stack(losses).transpose(0,1).transpose(2,1)
        context = torch.stack(context).transpose(0,1).transpose(2,1)

        ## remove padding tokens
        losses = losses.view(-1, 2 * b)[mask.flatten()]
        context = context.view(-1, 2 * b)[mask.flatten()]

        ffn_gate_all_layer = torch.stack(result['ffn_gate'])[:, :, :-1, :]
        l, bs, seq_len, h = ffn_gate_all_layer.shape
        ffn_gate_all_layer = ffn_gate_all_layer.reshape(l, bs * seq_len, h).transpose(0, 1)
        ffn_gate_all_layer = ffn_gate_all_layer[mask.flatten()]

        res = dict(context=context, loss=losses, ffn_gate=ffn_gate_all_layer)

        return res


    def get_act(self, dataset, cache_name, layers, acts, text_name='text', batch_size=16, start_batch=0, end_batch=5):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        cache_subdir = os.path.join(cache_dir, f'bs_{batch_size}_{start_batch}-{end_batch}')
        
        ## load from cache if possible
        if os.path.exists(cache_subdir):
            if 'context' not in acts.keys():
                with open(os.path.join(cache_subdir, 'context.pickle'), 'rb') as f:
                    acts['context'] = pickle.load(f)

            if 'loss' not in acts.keys():
                with open(os.path.join(cache_subdir, 'loss.pickle'), 'rb') as f:
                    acts['loss'] = pickle.load(f)

            for layer in layers:
                if f'layer_{layer}' not in acts.keys():
                    with open(os.path.join(cache_subdir, f'ffn_gate_{layer}.pickle'), 'rb') as f:
                        acts[f'layer_{layer}'] = pickle.load(f)
        
        ## if cache not exist, then create one
        else:
            os.makedirs(cache_subdir, exist_ok=True)

            acts = []

            for k in range(start_batch, end_batch):
                input = self.tokenizer(dataset[text_name][k * batch_size: (k + 1) * batch_size], return_tensors='pt', padding='longest')
                acts.append(self.get_neuron_activation_and_loss(input))

            context = torch.cat([i['context'] for i in acts], dim=0).numpy()
            loss = torch.cat([i['loss'] for i in acts], dim=0).numpy()
            ffn_gate = torch.cat([i['ffn_gate'] for i in acts], dim=0).numpy()

            del acts
            acts = dict(context=context, loss=loss)

            with open(os.path.join(cache_subdir, 'context.pickle'), 'wb') as f:
                pickle.dump(context, f)

            with open(os.path.join(cache_subdir, 'loss.pickle'), 'wb') as f:
                pickle.dump(loss, f)

            for layer in range(ffn_gate.shape[1]):
                with open(os.path.join(cache_subdir, f'ffn_gate_{layer}.pickle'), 'wb') as f:
                    acts[f'layer_{layer}'] = ffn_gate[:, layer, :]
                    pickle.dump(acts[f'layer_{layer}'], f)

        return acts

    def get_clm_loss(self, story, words, context_size, act_name='ffn_gate', context_token=True, use_cache=True, chunk = None):
        cache_file_name = f'{story}-context_size_{context_size}-clm_loss.pkl'
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        if use_cache and os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                ces = pickle.load(f)

        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
            context_array = self.get_story_array(words, context_size).cuda()
            total_size = context_array.size(0)

            if chunk is not None and chunk != 0:
                split_point = total_size // chunk
                res = []
                for context_array_part in torch.split(context_array, split_point):
                    res_part = ana.custom_forward(self.model, context_array_part, inspect_acts=[act_name])
                    logits = res_part['logits'].cuda()
                    labels = context_array_part.cuda()

                    shift_logits = logits[..., :-1, :].contiguous().view(-1, 32000)
                    shift_labels = labels[..., 1:].contiguous().view(-1)
                    
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels).cpu()
                    res.append(loss.view(len(res_part['logits']), -1))
                    del res_part

                ces = torch.cat(res).numpy()

            else:
                raise('NotImplement')

            with open(cache_file_path, 'wb') as f:
                    pickle.dump(ces, f)

        return ces

    def get_llm_act(self, story, words, context_size, act_name, layer, context_token=True, use_cache=True, chunk = None, cache_all_layer=True):
        cache_file_name = f'{story}-context_size_{context_size}-layer_{layer}-{act_name}-is_token_{context_token}.pkl'
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        if use_cache and os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                embs = pickle.load(f)

        else:
            context_array = self.get_story_array(words, context_size, context_token).cuda()
            # res = ana.custom_forward(self.model, context_array, inspect_acts=[act_name])
            # embs = res[act_name][layer][:, -1].numpy()
            # embs = torch.stack(res[act_name])[:, :, -1].numpy()
            total_size = context_array.size(0)

            if chunk is not None and chunk != 0:
                split_point = total_size // chunk
                res = []
                for context_array_part in torch.split(context_array, split_point):
                    res_part = ana.custom_forward(self.model, context_array_part, inspect_acts=[act_name])
                    embs_part_all_layer = torch.stack(res_part[act_name])[:, :, -1]
                    # print(embs_part_all_layer.shape)
                    del res_part
                    res.append(embs_part_all_layer)
                embs_all_layer = torch.cat(res, dim=1).numpy()
                

            else:
                context_array = self.get_story_array(words, context_size, context_token).cuda()
                res = ana.custom_forward(self.model, context_array, inspect_acts=[act_name])
                embs_all_layer = torch.stack(res[act_name])[:, :, -1].numpy()
            
            del res

            if cache_all_layer:
                for l in range(embs_all_layer.shape[0]):
                    cache_file_name = f'{story}-context_size_{context_size}-layer_{l}-{act_name}-is_token_{context_token}.pkl'
                    cache_file_path = os.path.join(self.cache_dir, cache_file_name)
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(embs_all_layer[l], f)
            else:
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(embs_all_layer[layer], f)

            embs = embs_all_layer[layer]

        return embs

    # def get_story_array(self, words, context_words):
    #     """get word ids for each phrase in a stimulus story
    #     """
    #     nctx = context_words + 1
    #     story_ids = self.encode(words)
    #     story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
    #     for i in range(len(story_array)):
    #         segment = story_ids[i:i+nctx]
    #         story_array[i, :len(segment)] = segment
    #     return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs