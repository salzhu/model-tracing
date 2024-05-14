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
# from unpermuted_mode_connectivity import unpermuted_mode_connectivity
from mode_connectivity_metrics import plot_traces, normalize_trace, max_loss, avg_loss, compute_p_value

import warnings
warnings.filterwarnings("ignore")

# print(os.environ['HF_HOME'])

block_size = 512

model_a_name = "meta-llama/Llama-2-7b-hf"
model_b_name = "lmsys/vicuna-7b-v1.5"

# model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
# model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

# model_temp = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)

# tokenizer_base = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# testset = load_filtered_dataset("dlwh/wikitext_103_detokenized", "test", tokenizer_base)

filename = "/nlp/u/salzhu/model-tracing/vicuna_llama_midpoint_wikitext_singlesample.csv"
plotpath = "temp"

# print("here")

for i in range(100):
    print("Mode connectivity test " + str(i) + "/100", end = " ")
    losses, perplexities = permuted_mode_connectivity(model_a_name, model_b_name, alpha_step=0.5)
    save_permuted_mode_connectivity_results(filename, plotpath, "llama2", "vicuna", 
                                            0.5, perplexities, losses, make_plots=False)

# unperm_loss, unperm_ppl = unpermuted_mode_connectivity(model_a_name, model_b_name, alpha_step=0.5)

# plot_traces(filename, "loss", "/nlp/scr/salzhu/vicuna_llama_midpoint.csv", "llama2", "vicuna", 
#             unpermuted_res=unperm_loss, alpha_step=0.5)

# count, total = max_loss(filename, unperm_loss, alpha_step=0.5)

# print("Max loss p-value: ")
# print(compute_p_value(count, total))
