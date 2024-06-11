import torch
from transformers import AutoModelForCausalLM
import csv
import os
import time
import scipy.stats

from utils import compute_emb_lap, match_emb, spcor
from permute import permute_embedding_layer

import warnings
warnings.filterwarnings("ignore")

block_size = 512

def emb_lap_permutations(model_a_name, model_b_name, filepath):

    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

    print("1")

    unperm_scor = compute_emb_lap(model_a,model_b)
    print(unperm_scor)

    # do scor to p-value

    # permute embedding layer
    # emb_permutation = torch.randperm(4096)
    # permute_embedding_layer(model_b,emb_permutation)

    perm_matched = match_emb(model_a,model_b)

    pvalue1 = scipy.stats.spearmanr(torch.arange(4096), perm_matched)

    random_scors = []
    for i in range(99):
        random_perm = torch.randperm(4096)
        random_scors.append(spcor(random_perm, perm_matched))

    count = 1
    total = len(random_scors) + 1

    for scor in random_scors:
        if scor < unperm_scor: count += 1

    pvalue2 = count / total
    print(pvalue1, pvalue2)

    model_a = model_a.to('cpu')
    model_b = model_b.to('cpu')
    del model_a, model_b
    

    csv_header = ["Model Pair", "unperm_scor", "p-value from spearman", "p-value from permutations"]

    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
    
    with open(filepath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, unperm_scor, pvalue1, pvalue2]
        writer.writerow(row)

file = "/nlp/u/salzhu/model-tracing/emb_lap_pvalues.csv"

model_list = [
        "meta-llama/Llama-2-7b-hf",
        "codellama/CodeLlama-7b-hf",
        "openlm-research/open_llama_7b",
        "huggyllama/llama-7b",
        "lmsys/vicuna-7b-v1.5",
        "EleutherAI/llemma_7b",
        "lmsys/vicuna-7b-v1.1",
        "microsoft/Orca-2-7b",
        "LLM360/Amber",
    ]

for i in range(len(model_list)):
    for j in range(i+1, len(model_list)):
        print("Starting...")
        time0 = time.time()
        print(model_list[i], model_list[j])
        emb_lap_permutations(model_list[i],  model_list[j], file)
        print("done! time was")
        print(str(time.time() - time0))

