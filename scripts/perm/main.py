import torch
import scipy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets
from tracing.utils.utils import normalize_mc_midpoint

from tracing.perm.permute import permute_model
from tracing.statistics.mc import statistic as mode_stat
from tracing.statistics.cossim_models import statistic as cossim_stat
from tracing.statistics.l2 import statistic as l2_stat

import timeit

from datasets import load_dataset, concatenate_datasets

MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

def p_value_exact(unpermuted, permuted): 
    count = 0
    for a in permuted:
        if(a < unpermuted):
            count += 1

    return round((count + 1) / (len(permuted) + 1), 2)

# def p_value_zscore(unpermuted, permuted):
#     all = permuted
#     all.append(unpermuted)
#     zscores = scipy.stats.zscore(all)
#     return zscores[-1]

def p_value_approx(unpermuted, permuted):
    mean = sum(permuted) / len(permuted)
    std = np.std(permuted)

    zscore = (unpermuted - mean) / std
    
    # return scipy.stats.norm.sf(abs(zscore))
    return zscore

def statistic(base_model, ft_model, mc_stat, l2_stat, num_perm, normalize=False):

    start = timeit.default_timer()

    unperm_stat_mc = mc_stat(base_model,ft_model)
    unperm_stat_l2 = l2_stat(base_model,ft_model)

    print(unperm_stat_mc, unperm_stat_l2, timeit.default_timer()-start)

    perm_stats_mc = []
    perm_stats_l2 = []

    for i in range(num_perm): 
        start = timeit.default_timer()

        mlp_permutation = torch.randperm(MLP_SIZE)
        emb_permutation = torch.randperm(EMB_SIZE)

        permute_model(ft_model, mlp_permutation, emb_permutation)

        perm_stat_mc = mc_stat(base_model, ft_model)
        perm_stat_l2 = l2_stat(base_model, ft_model)

        perm_stats_mc.append(perm_stat_mc)
        perm_stats_l2.append(perm_stat_l2)

        print(i, perm_stat_mc, perm_stat_l2, timeit.default_timer()-start)

    exact_mc = p_value_exact(unperm_stat_mc, perm_stats_mc.copy())
    approx_mc = p_value_approx(unperm_stat_mc, perm_stats_mc.copy())

    exact_l2 = p_value_exact(unperm_stat_l2, perm_stats_l2.copy())
    approx_l2 = p_value_approx(unperm_stat_l2, perm_stats_l2.copy())

    print(exact_mc, approx_mc)
    print(exact_l2, approx_l2)

    return exact_mc, approx_mc, exact_l2, approx_l2, unperm_stat_mc, unperm_stat_l2, perm_stats_mc, perm_stats_l2

def main(base_model, ft_model, test_stat, num_perm, normalize=False):

    base_metric = None
    ft_metric = None

    start = timeit.default_timer()

    # if normalize:
    #     base_metric = test_stat(base_model, base_model)
    #     ft_metric = test_stat(ft_model, ft_model)

    unperm_stat = test_stat(base_model,ft_model)

    # if normalize:
    #     unperm_stat = normalize_mc_midpoint(unperm_stat, base_metric, ft_metric)

    print(unperm_stat, timeit.default_timer()-start)

    perm_stats = []

    for i in range(num_perm): 
        start = timeit.default_timer()

        mlp_permutation = torch.randperm(MLP_SIZE)
        emb_permutation = torch.randperm(EMB_SIZE)

        permute_model(ft_model, mlp_permutation, emb_permutation)

        perm_stat = test_stat(base_model, ft_model)

        # if normalize:
        #     perm_stat = normalize_mc_midpoint(perm_stat, base_metric, ft_metric)

        perm_stats.append(perm_stat)
        print(i, perm_stat, timeit.default_timer()-start)

    # print(perm_stats)
    exact = p_value_exact(unperm_stat, perm_stats.copy())
    approx = p_value_approx(unperm_stat, perm_stats.copy())

    print(exact, approx)

    return exact, approx

if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    ft_model_name = "codellama/CodeLlama-7b-hf" # "lmsys/vicuna-7b-v1.1" # "huggyllama/llama-7b" 

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    # wikitext dataloader 
    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",N_BLOCKS,base_tokenizer)
    # dataloader = prepare_hf_dataloader(dataset,1)

    # m2d2 dataloader
    # json_file = "/juice4/scr4/nlp/model-tracing/m2d2_s2orc/AI/json_1.json"
    # columns_ignored = ['text', 'added', 'id', 'source', 'subdomain']
    # dataset = prepare_programming_dataset(json_file, N_BLOCKS, base_tokenizer, columns_ignored)
    # dataloader = prepare_hf_dataloader(dataset, 1)

    # generated dataloader
    dataset = load_generated_datasets(base_model_name, ft_model_name, 512, base_tokenizer, ["text"])
    dataloader = prepare_hf_dataloader(dataset, 1)


    # test_stat = lambda base_model,ft_model : mode_stat(base_model,ft_model,base_model,dataloader)
    # test_stat = lambda base_model,ft_model : cossim_stat(base_model,ft_model)
    test_stat = lambda base_model,ft_model : l2_stat(base_model,ft_model)

    main(base_model, ft_model, test_stat, 9)

    
