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

from utils import interpolate_models, evaluate, evaluate_one_batch
from permute import permute_model
from testsets import load_filtered_dataset

def permuted_mode_connectivity(model_base_name, model_ft_name, alpha_step=0.1):

    # Load base model, i.e. Llama2
    model_base = AutoModelForCausalLM.from_pretrained(model_base_name, torch_dtype=torch.bfloat16)
    tokenizer_base = AutoTokenizer.from_pretrained(model_base_name)

    # Load fine tuned model to permute
    model_ft = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=torch.bfloat16)

    testset = load_filtered_dataset("dlwh/wikitext_103_detokenized", "test", tokenizer_base)

    torch.manual_seed(datetime.datetime.now().timestamp())
    mlp_permutation = torch.randperm(11008)
    emb_permutation = torch.randperm(4096)

    permute_model(model_ft, mlp_permutation, emb_permutation)

    losses = []
    perplexities = []

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]

    for alpha in alphas:
        interpolated_model = interpolate_models(model_base, model_ft, alpha).to('cuda')
        loss, perplexity = evaluate_one_batch(interpolated_model, testset)
        perplexities.append(perplexity)
        losses.append(loss)
        interpolated_model.to("cpu")
        torch.cuda.empty_cache()
        print("alpha = " + str(alpha) + " | " + str(loss) + " | " + str(perplexity))

    return losses, perplexities

def save_permuted_mode_connectivity_results(csv_filename, plot_path, model_a_name, model_b_name, alpha_step, perplexities, losses, make_plots=True):

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]
    csv_header = ["Model Pair", "time"] + [f"Alpha {alpha} ppl" for alpha in alphas] + [f"Alpha {alpha} loss" for alpha in alphas]

    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, datetime.datetime.now()] + perplexities + losses
        writer.writerow(row)

    if make_plots:
        # Create the perplexity plot
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, perplexities)
        plt.xlabel("Alpha")
        plt.ylabel("Perplexity")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")

        # Save the plot as a PNG file
        plot_filename = f"{plot_path}perplexity/alpha_vs_perplexity_{model_a_name}_vs_{model_b_name}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        # Create the loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(alphas, losses)
        plt.xlabel("Alpha")
        plt.ylabel("Loss")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")

        # Save the plot as a PNG file
        plot_filename = f"{plot_path}loss/alpha_vs_losses_{model_a_name}_vs_{model_b_name}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()
        
# llama2_models_names = [("huggyllama/llama-7b", "huggyllama-7b"), ("openlm-research/open_llama_7b", "openllama-7b"), ("LLM360/Amber", "amber-7b"), ("meta-llama/Llama-2-7b-chat-hf", "llama2-chat-7b"), ("lmsys/vicuna-7b-v1.5", "vicuna-1.5-7b"), ("EleutherAI/llemma_7b", "llemma-7b"), ("codellama/CodeLlama-7b-hf", "codellama-7b"), ("lmsys/vicuna-7b-v1.1", "vicuna-1.1-7b")]

# base = "meta-llama/Llama-2-7b-hf"
# base_short_name = "llama2-7b"

# for model_b in llama2_models_names:
#     print("Mode connectivity on " + model_b[1])
#     losses, perplexities = permuted_mode_connectivity(base, model_b[0])

#     print("Losses: ", end = '')
#     print(losses)

#     print("Perplexities: ", end = '')
#     print(perplexities)

#     print("Saving results and plots...")
#     save_permuted_mode_connectivity_results("/nlp/scr/salzhu/permutation_mode_connectivity_tests.csv", "/nlp/scr/salzhu/permutation_mode_connectivity_plots/", 
#                                             base_short_name, model_b[1], 0.1, perplexities, losses)

#     print("Saved!")
