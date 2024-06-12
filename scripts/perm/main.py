import torch
import scipy
from transformers import AutoTokenizer, AutoModelForCausalLM
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader

from tracing.perm.permute import permute_model
from tracing.statistics.mc import statistic as mode_stat

MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

def p_value_exact(unpermuted, permuted): 
    count = 0
    for a in permuted:
        if(a < unpermuted):
            count += 1

    return round((count + 1) / (len(permuted) + 1), 2)

def p_value_zscore(unpermuted, permuted):
	all = permuted.append(unpermuted)
	zscores = scipy.stats.zscore(all)
	return zscores[-1]

def main(base_model, ft_model, test_stat, num_perm):

    unperm_stat = test_stat(base_model,ft_model)

    print("here")
    print(unperm_stat)

    perm_stats = []

    for i in range(num_perm): 
        mlp_permutation = torch.randperm(MLP_SIZE)
        emb_permutation = torch.randperm(EMB_SIZE)

        print("here1")

        permute_model(ft_model, mlp_permutation, emb_permutation)

        print("here2")

        perm_stat = test_stat(base_model, ft_model)
        perm_stats.append(perm_stat)
        print(perm_stat)

    p_value_1 = p_value_exact(unperm_stat, perm_stats)
    p_value_2 = p_value_zscore(unperm_stat, perm_stats)

    print(p_value_1, p_value_2)

    return p_value_1, p_value_2

if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    ft_model_name = "codellama/CodeLlama-7b-hf"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    print("here")

    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",N_BLOCKS,base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset,1)

    print("here")

    test_stat = lambda base_model,ft_model : mode_stat(base_model,ft_model,base_model,dataloader)

    print("here")

    main(base_model, ft_model, test_stat, 5)
