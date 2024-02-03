import torch
import os
import gc
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import random
from functools import partial
#from custom.modeling_llama import LlamaForCausalLM
from datasets.utils.logging import disable_progress_bar
import datasets
from typing import List, Mapping, NewType, Optional, Tuple, Union
import pickle
from collections import Counter
import math 
import time

disable_progress_bar()
#logging.disable_progress_bar()

# import utils.mmlu as mmlu
import utils_llama.modeling_llama as modeling_llama

# @torch.no_grad()
# def custom_forward(model, 
#                    input_ids, 
#                    early_exit = None, 
#                    inspect_acts = [], 
#                    forward_layer_ids = None, 
#                    skip_layer_ids = None, 
#                    record_layer_ids = None, 
#                    return_logits = False,
#                    add_attn = True,
#                    add_ffn = True,
#                    fake_act_args = None):
#     ### inspect_acts : ffn, attn, ffn_gate
    
#     assert forward_layer_ids is None or skip_layer_ids is None
    
#     activations = {i:[] for i in inspect_acts}

#     batch_size,  seq_length = input_ids.shape
#     seq_length_with_past = seq_length
#     device = input_ids.device
#     past_key_values_length = 0
#     position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).view(-1, seq_length)
#     inputs_embeds = model.model.embed_tokens(input_ids)
#     attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
#     attention_mask = model.model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
#     hidden_states = inputs_embeds
#     output_states = inputs_embeds
    
#     if forward_layer_ids is None:
#         forward_layer_ids = list(range(len(model.model.layers)))
#     if record_layer_ids is None:
#         record_layer_ids = list(range(len(model.model.layers)))
        
#     for idx in forward_layer_ids:
#         decoder_layer = model.model.layers[idx]

#         if idx == early_exit:
#             break
            
#         residual_attn = hidden_states
#         hidden_states = decoder_layer.input_layernorm(hidden_states)
#         hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=None,
#         )
#         if add_attn and idx in record_layer_ids:
#             output_states = output_states + hidden_states.to(output_states.device)
#         hidden_states = residual_attn.to(hidden_states.device) + hidden_states
        
#         residual_mlp = hidden_states
#         hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
#         if 'ffn_gate' in activations.keys():
#             gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states))
#             hidden_states = decoder_layer.mlp.down_proj(gates * decoder_layer.mlp.up_proj(hidden_states))
#             activations['ffn_gate'].append(gates.cpu())
#             del gates
#             torch.cuda.empty_cache()
#         else:
#             hidden_states = decoder_layer.mlp(hidden_states)
#         if 'ffn' in activations.keys():
#             activations['ffn'].append(hidden_states.cpu())
#         if add_ffn and idx in record_layer_ids:
#             output_states = output_states + hidden_states.to(output_states.device)
#         if skip_layer_ids is not None and idx in skip_layer_ids:
#             hidden_states = residual_attn
#         else:
#             hidden_states = residual_mlp + hidden_states
#         if skip_layer_ids is not None and idx == skip_layer_ids[-1]:
#             hidden_states = output_states
        
#         if fake_act_args is not None and fake_act_args['layer'] == idx:
#             pos = fake_act_args['pos']
#             hidden_states[0, pos] = fake_act_args['act'].to(hidden_states.device)
            
#     logits = None
#     if return_logits or 'logits' in activations.keys():
#         hidden_states = output_states
#         hidden_states = model.model.norm(hidden_states)
#         logits = model.lm_head(hidden_states).cpu()
#         activations['logits'] = logits
        
#     return activations, hidden_states, logits
  


# def inspect_activation(model, input_ids, inspect_layer = -1, inspect_ffn = False, forward_layer_ids = None):
#     activations = {'ffn':[], 'attn':[]}
    
#     if inspect_ffn:
#         activations['ffn_gate']=[]
    
#     batch_size,  seq_length = input_ids.shape
#     seq_length_with_past = seq_length
#     device = input_ids.device
#     past_key_values_length = 0
#     position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).view(-1, seq_length)
#     inputs_embeds = model.model.embed_tokens(input_ids)
#     attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
#     attention_mask = model.model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
#     hidden_states = inputs_embeds
    
#     if forward_layer_ids is None:
#         forward_layer_ids = list(range(len(model.model.layers)))
        
#     for idx in forward_layer_ids:
#         decoder_layer = model.model.layers[idx]
#     #for idx, decoder_layer in enumerate(model.model.layers):
#         if inspect_layer > 0 and idx == inspect_layer:
#             break

#         residual = hidden_states
#         hidden_states = decoder_layer.input_layernorm(hidden_states)
#         hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=None,
#         )
#         hidden_states = residual.to(hidden_states.device) + hidden_states
        
#         residual = hidden_states
#         hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
#         if inspect_ffn:
#             gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states))
#             hidden_states = decoder_layer.mlp.down_proj(gates * decoder_layer.mlp.up_proj(hidden_states))
#             activations['ffn_gate'].append(gates.cpu())
#             del gates
#             torch.cuda.empty_cache()
#         else:
#             hidden_states = decoder_layer.mlp(hidden_states)
#         activations['ffn'].append(hidden_states.cpu())
#         hidden_states = residual + hidden_states
#     return activations, hidden_states
  
class FFNProjector:
    def __init__(self, layer1, layer2, net):
        self.layer1 = layer1
        self.layer2 = layer2
        self.net = net
        self.act = None
        

@torch.no_grad()
def custom_forward(model, 
                   input_ids, 
                   position_ids = None,
                   past_key_values = None,
                   early_exit = None, 
                   inspect_acts = [], 
                   forward_layer_ids = None, 
                   skip_layer_ids = None, 
                   record_layer_ids = None, 
                   return_logits = True,
                   add_attn = None,
                   add_ffn = None,
                   fake_act_args = None,
                   use_cache = True,
                   draft_config = None,
                   fake_ffn = None,
                   fake_ffn_direct_contribution_only = False,
                   fake_ffn_neuron_filter = None,
                   fake_ffn_neuron_filter_zero = False,
                   return_all_act = False,
                   debug = False):

    # explain some hyperparameters (2024.2.1)
    # fake_ffn: Projector or List of Projectors
    # fake_ffn_direct_contribution_only: bool, whether to consider the indirect contribution of ffn activations
    # fake_ffn_neuron_filter: Dict of neuron indices for various layers, indicating positions that uses predicted values to replace real values.
    # fake_ffn_neuron_filter_zero: bool, replace the unfiltered neuron with zeros if True, otherwise real values
    ### inspect_acts : ffn, attn, ffn_gate
    # forward_layer_ids : layers that are selected as the small model
    # record_layer_ids : layers that directly contribute to the final logits
    
    assert forward_layer_ids is None or skip_layer_ids is None
    
    # support attn ffn ffn_gate layer_input

    # if return_all_act:
    activations = {i:[] for i in inspect_acts}

    batch_size,  seq_length = input_ids.shape
    seq_length_with_past = seq_length
    device = input_ids.device
    past_key_values_length = 0
    
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
        
    if position_ids is not None:
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    elif draft_config is not None and draft_config['tree_decoding'] > 1 and draft_config['token_dependency'] is not None:
        position_ids = torch.ones(seq_length, dtype=torch.long, device=device) * draft_config['position_id']
    else:
        position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    
    inputs_embeds = model.model.embed_tokens(input_ids)
    #print(input_ids)
    if draft_config is not None and draft_config['tree_decoding'] > 1 and draft_config['token_dependency'] is not None:
        token_dependency = draft_config['token_dependency']
        dependency_mask = torch.ones(batch_size, 1, seq_length, seq_length_with_past, dtype=torch.bool, device=inputs_embeds.device)
        total_length = len(token_dependency)
        for i in range(seq_length):
            p = total_length - seq_length + i
            pos = []
            while p != -1:
                pos.append(-total_length + p)
                p = token_dependency[p]
            dependency_mask[:, :, i, pos] = 0
            #print([len(token_dependency)+i for i in pos])
        dependency_mask[:, :, :, :-total_length] = 0
        # attention_mask = dependency_mask.to(inputs_embeds.dtype).masked_fill(dependency_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        
        attention_mask = dependency_mask.to(inputs_embeds.dtype).masked_fill(dependency_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        
    else:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
        attention_mask = model.model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
    

    hidden_states = inputs_embeds
    output_states = inputs_embeds
    
    if draft_config is None:
        if forward_layer_ids is None:
            forward_layer_ids = list(range(len(model.model.layers)))
        if record_layer_ids is None:
            record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_skip_uniform':
        i, j, s = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['step']
        forward_layer_ids = list(range(i)) + list(range(i, j, s)) + list(range(j, len(model.model.layers)))
        record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_dropout':
        i, j, p = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['prob']
        c = int(len(list(range(i, j))) * p)
        forward_layer_ids = list(range(i)) + random.sample(list(range(i, j)), c) + list(range(j, len(model.model.layers)))
        forward_layer_ids.sort()
        record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_dropout_ladder_1' or draft_config['mode'] == 'layer_dropout_ladder_2':
        i, j, s = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['step']
        k = draft_config['k']
        forward_layer_ids = list(range(i)) + list(range(i+k, j, s)) + list(range(j, len(model.model.layers)))
        record_layer_ids = list(range(len(model.model.layers)))
    else:
        if forward_layer_ids is None:
            forward_layer_ids = list(range(len(model.model.layers)))
        if record_layer_ids is None:
            record_layer_ids = list(range(len(model.model.layers)))
        
    next_decoder_cache = () if use_cache else None
    
    
    for idx in list(range(len(model.model.layers))):
        
        if idx not in forward_layer_ids:
            if use_cache:
                next_decoder_cache += (next_decoder_cache[-1],)
            continue
            
        decoder_layer = model.model.layers[idx]

        decoder_layer_self_attn = model.model.layers[idx].self_attn

        decoder_layer_mlp = model.model.layers[idx].mlp
        
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        if idx == early_exit:
            break

        if 'layer_input' in activations.keys():
            activations['layer_input'].append(hidden_states.cpu())
            
        residual_attn = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        
        if not debug:
            hidden_states, self_attn_weights, present_key_value = decoder_layer_self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                #output_attentions=True, 
                use_cache=use_cache,
                output_attentions=True

            )

            if 'attn' in activations.keys():
                activations['attn'].append(self_attn_weights.cpu())
        
        #################################
        ### customized self attention ###
        #################################
        else:
            sat = decoder_layer.self_attn
            bsz, q_len, _ = hidden_states.size()
                
            query_states = sat.q_proj(hidden_states).view(bsz, q_len, sat.num_heads, sat.head_dim).transpose(1, 2)
            key_states = sat.k_proj(hidden_states).view(bsz, q_len, sat.num_heads, sat.head_dim).transpose(1, 2)
            value_states = sat.v_proj(hidden_states).view(bsz, q_len, sat.num_heads, sat.head_dim).transpose(1, 2)
            
            if debug and idx == 0:
               # print(position_ids)
                activations['debug1'] = hidden_states.cpu()
                activations['debug2'] = sat.q_proj(hidden_states).cpu()
            
            
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = sat.rotary_emb(value_states, seq_len=kv_seq_len)
                
                
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            # [bsz, nh, t, hd]

            if past_key_value is not None:
                # reuse k, v, self_attention
                # print(past_key_value[0].device, past_key_value[0].shape)
                # print(past_key_value[0].device, past_key_value[0].shape)
                # print(key_states.device, key_states.shape)
 
                # key_states = torch.cat([past_key_value[0], key_states], dim=2)
                # value_states = torch.cat([past_key_value[1], value_states], dim=2)

                key_states = torch.cat([past_key_value[0].to(key_states.device), key_states], dim=2)
                value_states = torch.cat([past_key_value[1].to(key_states.device), value_states], dim=2)

            present_key_value = (key_states, value_states) if use_cache else None

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(sat.head_dim)
                
            if attn_weights.size() != (bsz, sat.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * sat.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            
          
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # print(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)

            if 'attn' in activations.keys():
                activations['attn'].append(attn_weights.cpu())
                
            if attn_output.size() != (bsz, sat.num_heads, q_len, sat.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, sat.num_heads, q_len, sat.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, sat.hidden_size)

            attn_output = sat.o_proj(attn_output)

            hidden_states = attn_output

        
        if use_cache:
            next_decoder_cache += (present_key_value,)
            
        if add_attn is None or (idx in add_attn):
        # if add_attn and idx in record_layer_ids:
            output_states = output_states + hidden_states.to(output_states.device)

        hidden_states = residual_attn.to(hidden_states.device) + hidden_states
        
        residual_mlp = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        
        
            
        if 'ffn_gate' in activations.keys():

            gates, fake_gates = None, None
            is_predicted = False

            if fake_ffn is None:
                gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)
            
            elif type(fake_ffn) is not list:
                if idx == fake_ffn.layer2:
                    fake_gates = fake_ffn.act
                    # gates = fake_gates
                    is_predicted = True
                # if idx != fake_ffn.layer2 or fake_ffn_direct_contribution_only:
                #     gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)
                gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)
                if idx == fake_ffn.layer1:
                    fake_ffn.act = fake_ffn.net.to(gates.device)(gates)

            else:
                l1toproj = {fake_ffn[i].layer1 : i for i in range(len(fake_ffn))}
                l2toproj = {fake_ffn[i].layer2 : i for i in range(len(fake_ffn))}
                if idx in l2toproj.keys():
                    assert fake_ffn[l2toproj[idx]].act is not None
                    fake_gates = fake_ffn[l2toproj[idx]].act
                    # gates = fake_gates
                    is_predicted = True
                # if idx not in l2toproj.keys() or fake_ffn_direct_contribution_only:
                #     gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)
                gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)
                if idx in l1toproj.keys():
                    fake_ffn[l1toproj[idx]].act = fake_ffn[l1toproj[idx]].net.to(gates.device)(gates)

            # print(l1toproj, l2toproj)
            # print(idx)

            # if fake_ffn is not None and idx == fake_ffn.layer2:
            #     gates = fake_ffn.act
            # else:
            #     # gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states))
            #     # hidden_states = decoder_layer.mlp.down_proj(gates * decoder_layer.mlp.up_proj(hidden_states))
            #     gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states)) * decoder_layer.mlp.up_proj(hidden_states)

            # if fake_ffn is not None and idx == fake_ffn.layer1:
            #     fake_ffn.act = fake_ffn.net.to(gates.device)(gates)

            # apply neuron filter to the fake_gates
            if fake_ffn_neuron_filter is not None and idx in fake_ffn_neuron_filter.keys() and fake_gates is not None:
                neuron_filter = fake_ffn_neuron_filter[idx].to(fake_gates.device)
                if fake_ffn_neuron_filter_zero:
                    fake_gates = fake_gates * neuron_filter
                else:
                    fake_gates = fake_gates * neuron_filter + gates * (~neuron_filter)
            
            # if consider the full contribution of fake ffn, then replace gates with fake_gates
            if not fake_ffn_direct_contribution_only and fake_gates is not None:
                gates = fake_gates

            fake_hidden_states = None
            if fake_ffn_direct_contribution_only and fake_gates is not None:
                fake_hidden_states = decoder_layer.mlp.down_proj(fake_gates)

            hidden_states = decoder_layer.mlp.down_proj(gates)
            activations['ffn_gate'].append(gates.cpu())
            del gates
            torch.cuda.empty_cache()
        else:
            hidden_states = decoder_layer_mlp(hidden_states)
        if 'ffn' in activations.keys():
            activations['ffn'].append(hidden_states.cpu())
        
        if add_ffn is None or (idx in add_ffn):

            # if fake_hidden_states is not None, then it means only direct contributions need to be considered
            if fake_hidden_states is not None:
                output_states = output_states + fake_hidden_states.to(output_states.device)
            else: 
                output_states = output_states + hidden_states.to(output_states.device)

        if skip_layer_ids is not None and idx in skip_layer_ids:
            hidden_states = residual_attn
        else:
            hidden_states = residual_mlp + hidden_states
        if skip_layer_ids is not None and idx == skip_layer_ids[-1]:
            hidden_states = output_states
        
        if fake_act_args is not None and fake_act_args['layer'] == idx:
            pos = fake_act_args['pos']
            hidden_states[0, pos] = fake_act_args['act'].to(hidden_states.device)
            
       
    next_cache = next_decoder_cache if use_cache else None
    # activations['past_key_values'] = next_cache
        
    logits = None
    if return_logits or 'logits' in activations.keys():
        hidden_states = output_states
        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states).cpu()
        activations['logits'] = logits.float()
        
    return activations
  
    
    
    
@torch.no_grad()
def custom_forward_dev(model, 
                   input_ids, 
                   position_ids = None,
                   past_key_values = None,
                   early_exit = None, 
                   inspect_acts = [], 
                   forward_layer_ids = None, 
                   skip_layer_ids = None, 
                   record_layer_ids = None, 
                   return_logits = True,
                   add_attn = True,
                   add_ffn = True,
                   fake_act_args = None,
                   use_cache = True,
                   draft_config = None,
                   self_speculative = False,
                   debug = False):
    ### inspect_acts : ffn, attn, ffn_gate
    # forward_layer_ids : layers that are selected as the small model
    # record_layer_ids : layers that directly contribute to the final logits
    
    assert forward_layer_ids is None or skip_layer_ids is None
    
    activations = {i:[] for i in inspect_acts}

    batch_size,  seq_length = input_ids.shape
    seq_length_with_past = seq_length
    device = input_ids.device
    past_key_values_length = 0
    
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
        
    if position_ids is not None:
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    elif draft_config is not None and draft_config['tree_decoding'] > 1 and draft_config['token_dependency'] is not None:
        position_ids = torch.ones(seq_length, dtype=torch.long, device=device) * draft_config['position_id']
    else:
        position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    
    inputs_embeds = model.model.embed_tokens(input_ids)
    #print(input_ids)
    if draft_config is not None and draft_config['tree_decoding'] > 1 and draft_config['token_dependency'] is not None:
        token_dependency = draft_config['token_dependency']
        dependency_mask = torch.ones(batch_size, 1, seq_length, seq_length_with_past, dtype=torch.bool, device=inputs_embeds.device)
        total_length = len(token_dependency)
        for i in range(seq_length):
            p = total_length - seq_length + i
            pos = []
            while p != -1:
                pos.append(-total_length + p)
                p = token_dependency[p]
            dependency_mask[:, :, i, pos] = 0
            #print([len(token_dependency)+i for i in pos])
        dependency_mask[:, :, :, :-total_length] = 0
        # attention_mask = dependency_mask.to(inputs_embeds.dtype).masked_fill(dependency_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        
        attention_mask = dependency_mask.to(inputs_embeds.dtype).masked_fill(dependency_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
        
    else:
        attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
        attention_mask = model.model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
    
    hidden_states = inputs_embeds
    output_states = inputs_embeds
    
    if draft_config is None:
        if forward_layer_ids is None:
            forward_layer_ids = list(range(len(model.model.layers)))
        if record_layer_ids is None:
            record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_skip_uniform':
        i, j, s = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['step']
        forward_layer_ids = list(range(i)) + list(range(i, j, s)) + list(range(j, len(model.model.layers)))
        record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_dropout':
        i, j, p = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['prob']
        c = int(len(list(range(i, j))) * p)
        forward_layer_ids = list(range(i)) + random.sample(list(range(i, j)), c) + list(range(j, len(model.model.layers)))
        forward_layer_ids.sort()
        record_layer_ids = list(range(len(model.model.layers)))
    elif draft_config['mode'] == 'layer_dropout_ladder_1' or draft_config['mode'] == 'layer_dropout_ladder_2':
        i, j, s = draft_config['bottom'], len(model.model.layers) - draft_config['top'], draft_config['step']
        k = draft_config['k']
        forward_layer_ids = list(range(i)) + list(range(i+k, j, s)) + list(range(j, len(model.model.layers)))
        record_layer_ids = list(range(len(model.model.layers)))
    else:
        if forward_layer_ids is None:
            forward_layer_ids = list(range(len(model.model.layers)))
        if record_layer_ids is None:
            record_layer_ids = list(range(len(model.model.layers)))
        
    next_decoder_cache = () if use_cache else None
    
    # print(forward_layer_ids)
    
    if self_speculative:
        ATTN_LAYERS, FFN_LAYERS = model.get_skip_layers()
        # ATTN_LAYERS = []#list(range(len(model.model.layers)))
        # FFN_LAYERS = []#list(range(len(model.model.layers)))
       
    t = 0
    for idx in list(range(len(model.model.layers))):
              
        decoder_layer = model.model.layers[idx]

        decoder_layer_self_attn = model.model.layers[idx].self_attn
        decoder_layer_mlp = model.model.layers[idx].mlp
        
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        
        if idx == early_exit:
            break
            
        residual = hidden_states
        
        
        if self_speculative and idx in ATTN_LAYERS:
            hidden_states = residual
            present_key_value = None
            
        else:
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            
            hidden_states, self_attn_weights, present_key_value = decoder_layer_self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # if idx == 0:
            #     print(attention_mask.shape)
            #     print(position_ids.shape)
            hidden_states = residual.to(hidden_states.device) + hidden_states
        
        if use_cache:
            next_decoder_cache += (present_key_value,)

        residual = hidden_states
        
        t1 = time.time()
        
        if self_speculative and idx in FFN_LAYERS:
            hidden_states = residual
            
        else:
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer_mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states
        t2 = time.time()
        t += t2-t1
        
                 
                    
    next_cache = next_decoder_cache if use_cache else None
    activations['past_key_values'] = next_cache
    
    #print('attn', t)
    logits = None
    if return_logits or 'logits' in activations.keys():
        hidden_states = output_states
        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states).cpu()
        activations['logits'] = logits.float()
        
    return activations
  
    
    
def custom_head_wo_norm(model, hidden_states):
    logits = model.lm_head(hidden_states)
    return logits

def custom_head(model, hidden_states):
    hidden_states = model.model.norm(hidden_states)
    return custom_head_wo_norm(model, hidden_states)


class LayerWeightModifier:
    def __init__(self, layer, module, channel=[]):
        self.layer = layer
        self.module = module
        self.channel = channel
        
    def get_module(self, model):
        # assert isinstance(model, LlamaForCausalLM)
        m = model.model.layers[self.layer]
        if self.module == 'q_proj':
            module = m.self_attn.q_proj
        elif self.module == 'k_proj':
            module = m.self_attn.k_proj
        elif self.module == 'v_proj':
            module = m.self_attn.v_proj
        elif self.module == 'o_proj':
            module = m.self_attn.o_proj
        elif self.module == 'gate_proj':
            module = m.mlp.gate_proj
        elif self.module == 'down_proj':
            module = m.mlp.down_proj
        elif self.module == 'up_proj':
            module = m.mlp.up_proj
        return module
    
    def do(self, model):
        if self.channel == None:
            return 
        module = self.get_module(model)
        module.requires_grad = False
        self.device = module.weight.device
        self.weight_backup = module.weight.data[self.channel].clone().cpu()
        out_features,  in_features = module.weight.shape
        # if self.channel == []:
        #     module.weight.data = torch.zeros(out_features, in_features, dtype=torch.float16).to(self.device)
        # else:
        #     module.weight.data[self.channel] = 0
        module.weight.data[self.channel] = 0
        
    def undo(self, model):
        if self.channel == None:
            return 
        module = self.get_module(model)
        module.weight.data[self.channel] = self.weight_backup.to(self.device)

class WeightModifier:
    def __init__(self, locations):
        ## locations : tuple of layer,module,channel triples
        self.modifiers = [LayerWeightModifier(i[0], i[1], i[2]) for i in locations]
    
    def apply(self, model):
        for m in self.modifiers:
            m.do(model)
    def unapply(self, model):
        for m in self.modifiers:
            m.undo(model)
        
        
        
class Activation:
    def __init__(self, model, prompt):
        self.prompt = prompt
        self.model = model
        self.input = tokenizer([prompt], return_tensors='pt')['input_ids']
        self.get_activation()
        
    def get_activation(self):
        self.activations = inspect_activation(self.model, self.input)['ffn']
        
    def origin(self):
        output = self.model(self.input, output_hidden_states=True)
        self.activations = output['hidden_states']
        self.logits = output['logits']
        
    def get(self, layer, token):
        t = self.activations[layer][0, token, :]
        print('layer:', layer, 'token:', token)
        print('top 10 values:', t.view(-1).sort(descending=True).values[:10].tolist())
        print('top 10 indices:', t.view(-1).sort(descending=True).indices[:10].tolist())

    def scale(self, layer):
        print('layer:', layer)
        print(self.activations[layer][0, :, :].max(axis=-1).values.tolist())
        
def evaluate_math(seed=1):
    random.seed(seed)
    cnt = 0 
    inputs = []
    answers = []
    for r in range(100):
        x = random.choice(list(range(100)))
        y = random.choice(list(range(100)))
        z = x * y
        prompt = str(x) + ' times ' + str(y) + ' equals '
        answers.append(z)
        inputs.append(prompt)
    outputs = generator(inputs, batch_size=50, max_length = 20)
    for i in range(100):
        z = answers[i]
        output = outputs[i][0]['generated_text']
        z1 = str(z)
        z2 = z1[:-3]+','+z1[-3:]
        if z1 in output or z2 in output:
            cnt += 1 
    return cnt

def extract_math(tokenizer, seed=1):
    random.seed(seed)
    cnt = 0 
    inputs = []
    answers = []
    actss =  []
    for r in range(100):
        x = random.choice(list(range(100)))
        y = random.choice(list(range(100)))
        z = x * y
        prompt = str(x) + ' times ' + str(y) + ' equals ' + str(z) 
        input_ids = tokenizer([prompt], return_tensors='pt').input_ids
        acts, hidden_states = inspect_activation(llama, input_ids, inspect_ffn=True)
        actss.append(acts['ffn_gate'])
    return actss

def extract_math_2(tokenizer, seed=1):
    random.seed(seed)
    cnt = 0 
    inputs = []
    answers = []
    actss =  []
    for r in range(100):
        x = random.choice(list(range(100)))
        y = random.choice(list(range(100)))
        z = x * y
        prompt = str(x) + ' times ' + str(y) + ' equals ' + str(z) 
        input_ids = tokenizer([prompt], return_tensors='pt').input_ids
        acts, hidden_states = inspect_activation(llama, input_ids, inspect_ffn=True)
        actss.append(acts['ffn_gate'])
    return actss

def extract_mmlu(tokenizer, model, category, bs = 10):
    actss =  []
    lengths = []
    _, test_df = mmlu.load_data(category)
    l = len(test_df[:100])//bs
    for i in range(l):
        input_data = tokenizer(test_df[0].tolist()[i*bs:i*bs+bs], return_tensors='pt', padding=True)
        with torch.no_grad():
            acts, hidden_states = inspect_activation(model, input_data.input_ids, inspect_ffn=True)
            actss.append(acts['ffn_gate'])
            lengths.append((input_data.attention_mask != 0).sum(axis=-1))
    #lengths = torch.stack(lengths).view(-1)
    return actss, lengths
        
def evaluate_generation():
    prompt = 'In a Utopian alternate universe, an author writes a sci-fi dystopian novel describing our society.'
    return generator(prompt, max_length = 100)

def create_simple_math(num=100, seed=1):
    random.seed(seed)
    inputs, outputs = [], []
    for i in range(num):
        x = random.choice(list(range(100)))
        y = random.choice(list(range(100)))
        z = x * y
        inputs.append(str(x) + ' times ' + str(y) + ' equals ') 
        outputs.append(str(z))
    return inputs, outputs

def extract_simple_math(tokenizer, model, bs=10, create_math_func=create_simple_math()):
    actss =  []
    lengths = []
    inputs, outputs = create_math_func()
    l = 100//bs
    for i in range(l):
        input_data = tokenizer(inputs[i*bs:i*bs+bs], return_tensors='pt', padding=True)
        with torch.no_grad():
            acts, hidden_states = inspect_activation(model, input_data.input_ids, inspect_ffn=True)
            actss.append(acts['ffn_gate'])
            lengths.append((input_data.attention_mask != 0).sum(axis=-1))
    return actss, lengths

def get_neuron_id_from_act(act, lengths, position='last'):
    ## topk : dict of topk activated neurons 
    k = 1000
    topk = {}
    topk['lengths'] = lengths
    if type(act[0]) == dict:
        tmp = [torch.stack(i['ffn_gate']).float().topk(k) for i in act]
    else:
        tmp = [torch.stack(i).float().topk(k) for i in act]
    topk['values'] = [i.values for i in tmp]
    topk['indices'] = [i.indices for i in tmp]
    
    if position == 'last':    
        ## layer x total_example x 1000
        value = torch.stack([torch.diagonal(topk['values'][batch][:, :, topk['lengths'][batch]-1, :], dim1=1,dim2=2).transpose(0,2) for batch in range(len(topk['lengths']))]).view(-1, k, 80).transpose(0,2).transpose(1,2)
        indice = torch.stack([torch.diagonal(topk['indices'][batch][:, :, topk['lengths'][batch]-1, :], dim1=1,dim2=2).transpose(0,2) for batch in range(len(topk['lengths']))]).view(-1, k, 80).transpose(0,2).transpose(1,2)
    elif position == 'avg':
        ## layer x total_length x 1000
        value = torch.cat([torch.cat([topk['values'][batch][:, i][:, :topk['lengths'][batch][i]] for i in range(5)], dim=1) for batch in range(len(topk['lengths']))], dim=1)
        indice = torch.cat([torch.cat([topk['indices'][batch][:, i][:, :topk['lengths'][batch][i]] for i in range(5)], dim=1) for batch in range(len(topk['lengths']))], dim=1)
    return value, indice

# def get_filterd_neuron_id(value, indice, value_thred=1, count_thred=0.8):
#     layers = indice.shape[0]
#     neurons = []
#     for i in range(layers):
#         ids = list(set(indice[i][value[i]>value_thred].tolist()))
#         neurons.append([i, ids])
#     return neurons


def get_filterd_neuron_id(value, indice, value_thred=1, count_thred=0):
    layers = indice.shape[0]
    total = indice.shape[1]
    neurons = []
    for i in range(layers):
        ids = indice[i][value[i]>value_thred].tolist()
        if count_thred == 0:
            ids = list(set(ids))
        else:
            t = []
            c = total * count_thred
            ids = [k for k, v in Counter(ids).items() if v >= c]
        neurons.append([i, ids])
    return neurons

### bug
def iterative_identify_activated_neuron(tokenizer, model, name, iter_count):
    wms = []
    identified_neurons = []
    for it in range(iter_count):
        actss, lengths = extract_mmlu(tokenizer, model, name)
        value, indice = get_neuron_id_from_act(actss, lengths)
        neurons = get_filterd_neuron_id(value, indice)
        wms.append(WeightModifier([[i[0], 'gate_proj', i[1]] for i in neurons]))
        wms[it].apply(model)
        identified_neurons.append(neurons)
    for wm in wms:
        wm.unapply(model)
    return wms, identified_neurons