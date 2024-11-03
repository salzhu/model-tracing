import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tracing.utils.evaluate import evaluate
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader
from tracing.statistics.mlp_sp import hook_out
from tracing.utils.evaluate import prepare_hf_dataset, prepare_hf_dataloader,evaluate, prepare_random_sample_dataset
from tracing.utils.llama.matching import match_wmats
from collections import defaultdict
from tracing.utils.utils import cossim

import scipy
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def mlp_matching_gate(base_model, ft_model, dataloader, i,j):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[j].mlp.gate_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    base_mat.to('cuda')
    ft_mat.to('cuda')

    print(ft_mat.shape)
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    print(ft_mat.shape)

    ft_mat = torch.norm(ft_mat,dim=1)
    print(ft_mat.shape)
    sorted = torch.sort(torch.argsort(ft_mat)[:8192])[0]
    for i in sorted:
        print(i.item(),end=" ")

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)

    # mat = cossim(ft_mat,base_mat)
    # perm = torch.argmax(mat,axis=-1)

    # Alternatively: Using cosine similarity matrix and taking argmax (does not work as well)
    """
    mat = cossim(base_mat,ft_mat)
    perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    """
    
    return perm

def mlp_matching_up(base_model, ft_model, dataloader, i,j):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.up_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[j].mlp.up_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    base_mat.to('cuda')
    ft_mat.to('cuda')
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)

    # Alternatively: Using cosine similarity matrix and taking argmax (does not work as well)
    """
    mat = cossim(base_mat,ft_mat)
    perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    """
    
    return perm


def mlp_layers(base_model,ft_model,dataloader,i,j):

    gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i,j)
    up_match = mlp_matching_up(base_model, ft_model, dataloader, i,j)

    print(len(gate_match))

    for g in gate_match:
        print(g.item(),end=' ')

    cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
    # print(i, pvalue, len(gate_match))

    return pvalue

def main():
    model_1_id = "meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-3.2-3B"
    model_2_id = "princeton-nlp/Sheared-LLaMA-2.7B" # "meta-llama/Llama-3.1-8B"

    print(model_1_id, model_2_id)

    model_1 = AutoModelForCausalLM.from_pretrained(model_1_id, torch_dtype=torch.bfloat16)
    model_2 = AutoModelForCausalLM.from_pretrained(model_2_id, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_1_id)

    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,tokenizer) 
    # dataloader = prepare_hf_dataloader(dataset,1)

    dataset = prepare_random_sample_dataset(20, 512)
    dataloader = prepare_hf_dataloader(dataset, 512)

    print(model_1.config.num_hidden_layers, model_2.config.num_hidden_layers)

    model_1_matched = np.zeros(model_1.config.num_hidden_layers)
    model_2_matched = np.zeros(model_2.config.num_hidden_layers)

    for i in range(model_1.config.num_hidden_layers):
        for j in range(model_2.config.num_hidden_layers):
            if model_1_matched[i] == 1 or model_2_matched[j] == 1:
                continue
            stat = mlp_layers(model_1, model_2, dataloader, i, j)
            print(i, j, stat)
            if(stat < 0.000001):
                model_1_matched[i] = 1
                model_2_matched[j] = 1
                break
            break
        break

if __name__ == "__main__":
    main()