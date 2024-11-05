import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tracing.statistics.mlp_sp import statistic
import numpy as np
import scipy
from tracing.utils.llama.matching import match_wmats
from collections import defaultdict
from tracing.utils.evaluate import evaluate, prepare_random_sample_dataset, prepare_hf_dataloader

def statistic(model_1, model_2, dataloader):
    stats = []
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

def mlp_layers(base_model,ft_model,dataloader,i,j):

    gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i,j)
    up_match = mlp_matching_up(base_model, ft_model, dataloader, i,j)

    cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())

    return pvalue

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def mlp_matching_gate(base_model, ft_model, dataloader, i, j):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    base_handle = None
    ft_handle = None

    if str(base_model.config.architectures[0]) in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", 
                                                   "Gemma2ForCausalLM", "GemmaForCausalLM"]:
        base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)
    elif str(base_model.config.architectures[0]) in ["StripedHyenaModelForCausalLM"]:
        base_handle = base_model.backbone.blocks[i].mlp.l1.register_forward_hook(base_hook)     

    if str(ft_model.config.architectures[0]) in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", 
                                                   "Gemma2ForCausalLM", "GemmaForCausalLM"]:
        ft_handle = ft_model.model.layers[j].mlp.gate_proj.register_forward_hook(ft_hook)
    elif str(ft_model.config.architectures[0]) in ["StripedHyenaModelForCausalLM"]:
        ft_handle = ft_model.backbone.blocks[j].mlp.l1.register_forward_hook(ft_hook)          
    
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
    
    return perm

def mlp_matching_up(base_model, ft_model, dataloader, i, j):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    base_handle = None
    ft_handle = None

    if str(base_model.config.architectures[0]) in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", 
                                                   "Gemma2ForCausalLM", "GemmaForCausalLM"]:
        base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)
    elif str(base_model.config.architectures[0]) in ["StripedHyenaModelForCausalLM"]:
        base_handle = base_model.backbone.blocks[i].mlp.l1.register_forward_hook(base_hook)     

    if str(ft_model.config.architectures[0]) in ["LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", 
                                                   "Gemma2ForCausalLM", "GemmaForCausalLM"]:
        ft_handle = ft_model.model.layers[j].mlp.gate_proj.register_forward_hook(ft_hook)
    elif str(ft_model.config.architectures[0]) in ["StripedHyenaModelForCausalLM"]:
        ft_handle = ft_model.backbone.blocks[j].mlp.l1.register_forward_hook(ft_hook)    
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
    
    return perm


def main():

    model_1 = AutoModelForCausalLM.from_pretrained("google/gemma-2b", 
                                                 torch_dtype=torch.bfloat16)
    model_2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", 
                                                 torch_dtype=torch.bfloat16)

    dataset = prepare_random_sample_dataset(20, 512)
    dataloader = prepare_hf_dataloader(dataset, 512)
    
    statistic(model_1,model_2,dataloader)

    # weights = model.state_dict()

    # layers_base = list(model.state_dict().keys())

    # for i in range(15):
    #     print(layers_base[i], weights[layers_base[i]].shape)

    # print("---------------------------------")

if __name__ == "__main__":
    main()