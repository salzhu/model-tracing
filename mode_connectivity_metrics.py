import torch
import math
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def plot_traces(results_path, metric, plot_path, model_a_name, model_b_name, unpermuted_res=False, normalize=True, alpha_step = 0.1):
    
    df = pd.read_csv(results_path)

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]
    
    if metric == 'loss':
        
        plt.figure(figsize=(8, 6))
        for index, row in df.iterrows():
            row = row[-int(1/alpha_step)-1:]
            if normalize: row=normalize_trace(row, alpha_step)
            plt.plot(alphas, row)
            
        plt.xlabel("Alpha")
        plt.ylabel("Loss")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
        plot_filename = f"{plot_path}alpha_vs_loss_{model_a_name}_vs_{model_b_name}_{datetime.datetime.now().timestamp()}.png"

    if metric == 'perplexity':
        
        plt.figure(figsize=(8, 6))
        for index, row in df.iterrows():
            row = row[-int(1/alpha_step)-1:]
            if normalize: row=normalize_trace(row, alpha_step)
            plt.plot(alphas, row)
            
        plt.xlabel("Alpha")
        plt.ylabel("Perplexity")
        plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
        plot_filename = f"{plot_path}alpha_vs_ppl_{model_a_name}_vs_{model_b_name}_{datetime.datetime.now().timestamp()}.png"

    if unpermuted_res != False:
        plt.plot(alphas, normalize_trace(unpermuted_res, alpha_step))

    # Save the plot as a PNG file
    
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

def normalize_trace(trace, alpha_step):
    slope = trace[-1] - trace[0]
    start = trace[0]
    for i in range(len(trace)):
        trace[i] -= slope * alpha_step * i
        trace[i] -= start
    return trace


def max_loss(results_path, unpermuted_loss, normalize=True, alpha_step=0.1):
    df = pd.read_csv(results_path)
    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]

    permuted_max_losses = []

    for index, row in df.iterrows():
        row = row[-int(1/alpha_step)-1:]
        if normalize: row=normalize_trace(row, alpha_step)
        permuted_max_losses.append(max(row))

    unpermuted_max_loss = max(unpermuted_loss)

    counter = 0
    for m in permuted_max_losses:
        if(m > unpermuted_max_loss): counter += 1

    return counter, len(permuted_max_losses)

def avg_loss(results_path, unpermuted_loss, normalize=True, alpha_step=0.1):
    df = pd.read_csv(results_path)
    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]

    permuted_avg_losses = []

    for index, row in df.iterrows():
        row = row[-int(1/alpha_step)-1:]
        if normalize: row=normalize_trace(row, alpha_step)
        permuted_avg_losses.append(sum(row) / len(row))

    unpermuted_avg_loss = sum(unpermuted_loss) / len(unpermuted_loss)

    counter = 0
    for m in permuted_avg_losses:
        if(m > unpermuted_avg_loss): counter += 1

    return counter, len(permuted_avg_losses)

def compute_p_value(counter, total):
    return (total - counter + 1) / total

vicuna_unpermuted_loss = [5.366623401641846, 5.986485004425049, 6.671814441680908, 6.927042484283447, 7.014814376831055, 7.035386562347412, 7.0459303855896, 7.057350158691406, 7.077110290527344, 7.108644008636475, 7.143476963043213]

# Plotting without the unpermuted trace
# plot_traces("/nlp/scr/salzhu/vicuna_reseed_permutation_tests.csv", 'loss', 
#             '/nlp/scr/salzhu/permutation_mode_connectivity_plots/aggregate/',
#             "llama2-7b", "vicuna-1.5-7b")

# Plotting with the unpermuted trace
# plot_traces("/nlp/scr/salzhu/vicuna_reseed_permutation_tests.csv", 'loss', 
#             '/nlp/scr/salzhu/permutation_mode_connectivity_plots/aggregate/',
#             "llama2-7b", "vicuna-1.5-7b", unpermuted_res=vicuna_unpermuted_loss)

# How the unpermuted max loss compares to permuted max loss
print(max_loss("/nlp/scr/salzhu/vicuna_reseed_permutation_tests.csv", vicuna_unpermuted_loss))

# How the unpermuted avg loss (auc) compares to permuted avg loss
print(avg_loss("/nlp/scr/salzhu/vicuna_reseed_permutation_tests.csv", vicuna_unpermuted_loss))

print("Done!")
