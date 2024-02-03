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


    def get_llm_act(self, story, words, context_size, act_name, layer, context_token=True, use_cache=True):
        cache_file_name = f'{story}-context_size_{context_size}-layer_{layer}-{act_name}-is_token_{context_token}.pkl'
        cache_file_path = os.path.join(self.cache_dir, cache_file_name)
        if use_cache and os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                embs = pickle.load(f)

        else:
            context_array = self.get_story_array(words, context_size, context_token).cuda()
            res = ana.custom_forward(self.model, context_array, inspect_acts=[act_name])
            embs = res[act_name][layer][:, -1].numpy()
            # embs = torch.stack(res[act_name])[:, :, -1].numpy()
            with open(cache_file_path, 'wb') as f:
                pickle.dump(embs, f)

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