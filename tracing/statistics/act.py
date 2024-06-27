import torch
from collections import defaultdict
import scipy
import numpy as np
from scipy.stats import chi2

from transformers import AutoModelForCausalLM, AutoTokenizer

import os

from tracing.utils.utils import cossim
from tracing.utils.evaluate import evaluate
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets

N_BLOCKS = 32

# from huggingface_hub import login
# hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]

# login(token=hf_token)

def statistic(base_model, ft_model, dataloader):
    return act_spcor(base_model, ft_model, dataloader)

def hook(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def act_median_max(base_model,ft_model,dataloader,i):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,4096).T
    ft_mat = torch.vstack(feats['ft']).view(-1,4096).T
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)

def act_spcor_layer(base_model, ft_model, dataloader, i):

    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,4096).T
    ft_mat = torch.vstack(feats['ft']).view(-1,4096).T

    matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    # print(i, cor, pvalue)
    return pvalue

# def act_spcor(model1, model2, dataloader):
#     chi_squared = 0
#     num_layers = 32

#     for i in range(num_layers):
#         temp = act_spcor_layer(model1, model2, dataloader, i)
#         chi_squared -= 2 * np.log(temp)
#         print(i, temp) 

#     p_value = chi2.sf(chi_squared, df=2*num_layers)
#     return p_value

def act_spcor(base_model, ft_model, dataloader):

    chi_squared = 0
    num_layers = 32

    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(32):
        # base_hooks[f"{i}"] = lambda *args : hook(*args,feats,f"base_{i}")
        # ft_hooks[f"{i}"] = lambda *args : hook(*args,feats,f"ft_{i}")
        # base_model.model.layers[i].input_layernorm.register_forward_hook(base_hooks[f"{i}"])
        # ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hooks[f"{i}"])

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].input_layernorm.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"ft_"+layer)
        ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hooks[layer])


    # for i in range(32):
    #     base_model.model.layers[i].input_layernorm.register_forward_hook(lambda *args : hook(*args,feats,f"base_{i}"))
    #     ft_model.model.layers[i].input_layernorm.register_forward_hook(lambda *args : hook(*args,feats,f"ft_{i}"))

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    p_values = []

    for i in range(32):

        base_mat = torch.vstack(feats[f'base_{i}']).view(-1,4096).T
        ft_mat = torch.vstack(feats[f'ft_{i}']).view(-1,4096).T

        matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
        orig = torch.arange(len(matched))

        cor, temp = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
        # print(i, cor, temp)
        chi_squared -= 2 * np.log(temp)
        p_values.append(temp)

    p_value = chi2.sf(chi_squared, df=2*num_layers)

    return p_value, p_values


if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    ft_model_name = "codellama/CodeLlama-7b-hf" # "lmsys/vicuna-7b-v1.1"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer)
    # dataloader = prepare_hf_dataloader(dataset,1)

    dataset = load_generated_datasets(base_model_name, ft_model_name, 512, base_tokenizer, ["text"])
    dataloader = prepare_hf_dataloader(dataset, 1)

    print(act_spcor(base_model, ft_model, dataloader))

    # print(statistic_spcor_all(base_model, ft_model, dataloader))