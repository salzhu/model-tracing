import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
import csv
import os
import os
import time

from permutation_tests import permuted_mode_connectivity_l2
from unpermuted_mode_connectivity import unpermuted_mode_connectivity
from l2_norm import calculate_l2_distance

import warnings
warnings.filterwarnings("ignore")

os.environ['HF_HOME'] = '/nlp/scr/salzhu/hf'
os.environ['PIP_CACHE_DIR'] = "/scr/salzhu/"
os.environ['WANDB_CACHE_DIR'] = "/scr/salzhu/"

print(os.environ['HF_HOME'])


block_size = 512

def permutations_main(model_a_name, model_b_name, num_perm, alpha_step, filepath_loss, filepath_norm_loss, filepath_l2):

    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

    tokenizer_base = AutoTokenizer.from_pretrained(model_a_name)

    losses = []
    l2s = []

    # unpermuted model metrics
    unperm_l2, a, b = calculate_l2_distance(model_a, model_b)
    unperm_losses, unperm_ppl = unpermuted_mode_connectivity(model_a_name, model_b_name, alpha_step=alpha_step, endpoints=True)
    unperm_loss = unperm_losses[1]
    print(unperm_l2, unperm_loss, flush=True)

    # apply permutations and metrics
    for i in range(num_perm):
        print("Mode connectivity test " + str(i) + "/" + str(num_perm), flush=True)
        loss, l2 = permuted_mode_connectivity_l2(model_a, model_b, tokenizer_base, alpha_step=alpha_step, end_points=False)
        losses.append(loss[0])
        l2s.append(l2[0])
    
    # computing normalized losses (level endpoints to 0)
    norm_losses = []
    for loss in losses:
        norm_losses.append(loss - (unperm_losses[0] + unperm_losses[2]) / 2)

    # losses, unnormalized p-value and write to file
    count = 0
    total = len(losses)
    for l in losses[:-1]:
        if unperm_loss <= l: count += 1
        
    loss_p = round(1 - count / total, 2)
    print("loss p-value: " + str(loss_p))

    csv_header = ["Model Pair", "loss p-value", "unpermuted loss"] + ["permuted losses"]

    if not os.path.exists(filepath_loss):
        with open(filepath_loss, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
    
    with open(filepath_loss, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, loss_p, unperm_loss] + losses
        writer.writerow(row)

    # losses, normalized p-value and write to file
    count = 0
    total = len(norm_losses)  

    unperm_norm_loss = unperm_losses[1] - (unperm_losses[0] + unperm_losses[2]) / 2
    for l in norm_losses[:-1]:
        if unperm_norm_loss <= l: count += 1
        
    norm_loss_p = round(1 - count / total, 2)
    print("norm loss p-value: " + str(norm_loss_p))

    csv_header = ["Model Pair", "norm loss p-value", "unpermuted norm loss"] + ["permuted norm losses"]

    if not os.path.exists(filepath_norm_loss):
        with open(filepath_norm_loss, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
    
    with open(filepath_norm_loss, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, norm_loss_p, unperm_norm_loss] + norm_losses
        writer.writerow(row)

    # l2 p-value and write to file
    count = 0
    total = len(l2s)

    for l in l2s[:-1]:
        if unperm_l2 <= l: count += 1
        
    l2_p = round(1 - count / total, 2)
    print("l2 p-value: " + str(l2_p))    

    csv_header = ["Model Pair", "l2 p-value", "unpermuted l2 dist"] + ["permuted l2 distances"]

    if not os.path.exists(filepath_l2):
        with open(filepath_l2, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
    
    with open(filepath_l2, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, l2_p, unperm_l2] + l2s
        writer.writerow(row)

file1 = "/nlp/u/salzhu/model-tracing/permutation_loss_midpoint_wikitext_single.csv"
file2 = "/nlp/u/salzhu/model-tracing/permutation_norm_loss_midpoint_wikitext_single.csv"
file3 = "/nlp/u/salzhu/model-tracing/permutation_l2_updated_midpoint_wikitext_single.csv"

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
        if(i == 0): continue
        if(i == 1): continue
        if(i == 2 and j <= 4): continue
        print("Starting...")
        time0 = time.time()
        print(model_list[i], model_list[j])
        permutations_main(model_list[i], model_list[j], 100, 0.5, file1, file2, file3)
        print("done! time was")
        print(str(time.time() - time0))

