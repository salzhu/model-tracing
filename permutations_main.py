import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import math
import numpy as np
import copy
import csv
import os
import matplotlib.pyplot as plt
import datetime
import os

from utils import interpolate_models
from permute import permute_model
from testsets import load_filtered_dataset
from permutation_tests import permuted_mode_connectivity, save_permuted_mode_connectivity_results
from unpermuted_mode_connectivity import unpermuted_mode_connectivity
from mode_connectivity_metrics import plot_traces, normalize_trace, max_loss, avg_loss, compute_p_value

import warnings
warnings.filterwarnings("ignore")

# print(os.environ['HF_HOME'])


block_size = 512

def permutations_main(model_a_name, model_b_name, num_perm, alpha_step, endpoints, metric, normalize, filepath, save_plot):

    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

    tokenizer_base = AutoTokenizer.from_pretrained(model_a_name)

    for i in range(num_perm):
        print("Mode connectivity test " + str(i) + "/" + str(num_perm), end = " ")
        losses, perplexities = permuted_mode_connectivity(model_a, model_b, tokenizer_base, alpha_step=alpha_step, end_points=endpoints)
        save_permuted_mode_connectivity_results(filepath, save_plot, model_a_name, model_b_name, 
                                                alpha_step, perplexities, losses, make_plots=False, end_points=endpoints)

    unperm_loss, unperm_ppl = unpermuted_mode_connectivity(model_a_name, model_b_name, alpha_step=alpha_step, endpoints=endpoints)

    if save_plot != None:
        plot_traces(filepath, "loss", save_plot, model_a_name, model_b_name, 
                unpermuted_res=unperm_loss, normalize=normalize, alpha_step=alpha_step, end_points=endpoints)

    count = 0
    total = 0
    
    if metric == 'max_loss':
        count, total = max_loss(filepath, unperm_loss, alpha_step=alpha_step)
    elif metric == 'avg_loss':
        count, total = avg_loss(filepath, unperm_loss, alpha_step=alpha_step)

    print("p-value: ")
    print(compute_p_value(count, total))

permutations_main("meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b", 10, 0.5, False,
                  "loss", False, "/nlp/u/salzhu/model-tracing/llama2_llama1_midpoint_wikitext_singlesample.csv",
                  None)