import yaml
from yaml import load, Loader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import json

base = {
   "lmsys/vicuna-7b-v1.5": 1, 
   "codellama/CodeLlama-7b-hf": 1, 
   "codellama/CodeLlama-7b-Python-hf": 1, 
   "codellama/CodeLlama-7b-Instruct-hf": 1,
   "EleutherAI/llemma_7b": 1,
   "microsoft/Orca-2-7b": 1,
   "oh-yeontaek/llama-2-7B-LoRA-assemble": 1,
   "lvkaokao/llama2-7b-hf-instruction-lora": 1,
   "NousResearch/Nous-Hermes-llama-2-7b": 1,
   "lmsys/vicuna-7b-v1.1": 0, 
   "yahma/llama-7b-hf": 0,
   "Salesforce/xgen-7b-4k-base": 2, 
   "EleutherAI/llemma_7b_muinstruct_camelmath": 1,
   "AlfredPros/CodeLlama-7b-Instruct-Solidity": 1,
   "meta-llama/Llama-2-7b-hf": 1, 
   "LLM360/Amber": 3, 
   "LLM360/AmberChat": 3,
   "openlm-research/open_llama_7b": 4,
   "openlm-research/open_llama_7b_v2": 5,
   "ibm-granite/granite-7b-base": 6,
   "ibm-granite/granite-7b-instruct": 6
}

base_ordered = {
    "yahma/llama-7b-hf": 0,
    "lmsys/vicuna-7b-v1.1": 0, 
    "meta-llama/Llama-2-7b-hf": 1, 
    "lmsys/vicuna-7b-v1.5": 1, 
    "codellama/CodeLlama-7b-hf": 1, 
    "codellama/CodeLlama-7b-Python-hf": 1, 
    "codellama/CodeLlama-7b-Instruct-hf": 1,
    "AlfredPros/CodeLlama-7b-Instruct-Solidity": 1,
    "EleutherAI/llemma_7b": 1,
    "EleutherAI/llemma_7b_muinstruct_camelmath": 1,
    "microsoft/Orca-2-7b": 1,
    "oh-yeontaek/llama-2-7B-LoRA-assemble": 1,
    "lvkaokao/llama2-7b-hf-instruction-lora": 1,
    "NousResearch/Nous-Hermes-llama-2-7b": 1,
    "Salesforce/xgen-7b-4k-base": 2, 
    "LLM360/Amber": 3, 
    "LLM360/AmberChat": 3,
    "openlm-research/open_llama_7b": 4,
    "openlm-research/open_llama_7b_v2": 5,
    "ibm-granite/granite-7b-base": 6,
    "ibm-granite/granite-7b-instruct": 6
}

tree = {
   "yahma/llama-7b-hf": "A---",
   "lmsys/vicuna-7b-v1.1": "AA--", 
   "meta-llama/Llama-2-7b-hf": "B---", 
   "lmsys/vicuna-7b-v1.5": "BA--", 
   "codellama/CodeLlama-7b-hf": "BB--", 
   "codellama/CodeLlama-7b-Python-hf": "BBA-", 
   "codellama/CodeLlama-7b-Instruct-hf": "BBB-",
   "AlfredPros/CodeLlama-7b-Instruct-Solidity": "BBBA",
   "EleutherAI/llemma_7b": "BBC-",
   "EleutherAI/llemma_7b_muinstruct_camelmath": "BBCA",
   "microsoft/Orca-2-7b": "BC--",
   "oh-yeontaek/llama-2-7B-LoRA-assemble": "BD--",
   "lvkaokao/llama2-7b-hf-instruction-lora": "BE--",
   "NousResearch/Nous-Hermes-llama-2-7b": "BF--",
   "Salesforce/xgen-7b-4k-base": "C---", 
   "LLM360/Amber": "D---", 
   "LLM360/AmberChat": "DA--",
   "openlm-research/open_llama_7b": "E---",
   "openlm-research/open_llama_7b_v2": "F---",
   "ibm-granite/granite-7b-base": "G---",
   "ibm-granite/granite-7b-instruct": "GA--"
}

def get_dict_ft(flat_model_path):
    dict_ft = {}

    model_paths = yaml.load(open(flat_model_path, 'r'), Loader=Loader)

    for i in range(len(model_paths)):
        for j in range(i+1,len(model_paths)):
            model_a = model_paths[i]
            model_b = model_paths[j]

            job_id = model_a.replace("/","-") + "_AND_" + model_b.replace("/","-")

            dict_ft[job_id] = (base[model_a] == base[model_b])

    return dict_ft

def get_statistic_from_file(filename):

    file = open(filename, 'r')

    lines = file.readlines()
    stat = np.nan

    for line in lines:
        if "Namespace" in line and "non-aligned test stat" in line:
            # dict = json.loads(line)
            # print(dict)
            # print(dict['non-aligned test stat'])

            start1 = line.find('non-aligned test stat')
            stat = line[line.find(':',start1):line.find(',',start1)]
            stat = stat.replace(' ', '')
            stat = stat.replace('(', '')
            stat = stat.replace(':', '')
            stat = float(stat)

    return stat


def plot_statistic_scatter(results_path, dict_ft, plot_path):

    x = []
    y = []

    dir_list = os.listdir(results_path)
    for file in dir_list:
        models = file[:file.find('.out')]
        if("huggyllama" in models): continue
        print(models)
        ft = int(dict_ft[models])
        stat = get_statistic_from_file(results_path + '/' + file)
        if not np.isnan(stat):
            x.append(ft)
            y.append(get_statistic_from_file(results_path + '/' + file))

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, s=10)
            
    plt.xlabel("Fine tuned")
    plt.ylabel("Test statistic")
    # plt.title(f"{}")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_statistic_grid(results_path, dict_base, plot_path, decimals, log=False):
    models = list(dict_base.keys())
    print(models)

    data = np.full((len(models), len(models)), np.nan)

    for i in range(len(models)):
        for j in range(len(models)):
            model_a = models[i]
            model_b = models[j]

            job_id = model_a.replace("/","-") + "_AND_" + model_b.replace("/","-") + ".out"

            if not os.path.exists(results_path + '/' + job_id): continue

            stat = get_statistic_from_file(results_path + '/' + job_id)

            if log: 
                stat = np.log(stat)
            
            data[i][j] = np.round(stat,decimals=decimals)
            data[j][i] = data[i][j]

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    im = ax.imshow(data)

    divider = make_axes_locatable(ax)
    # cbar = divider.append_axes("right", size="5%", pad=0.05)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.colorbar(im, cax=cbar)
    cbar.ax.set_ylabel("test statistic", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(models)), labels=models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Test statistic for model pairs")
    fig.tight_layout()
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    plt.close()

def plot_statistic_grid_bad(results_path, dict_ft, dict_base, plot_path, decimals):
    # model 1 vs model 2 
    # color the model names 

    models = list(dict_base.keys())
    print(models)

    data = np.full((len(models), len(models)), np.nan)

    for i in range(len(models)):
        for j in range(i+1,len(models)):
            model_a = models[i]
            model_b = models[j]

            job_id = model_a.replace("/","-") + "_AND_" + model_b.replace("/","-") + ".out"
            stat = get_statistic_from_file(results_path + '/' + job_id)
            data[i][j] = np.round(stat,decimals=decimals)

    cell_colors = plt.cm.viridis(data)

    model_labels = np.empty(len(models))
    for i in range(len(models)):
        model_labels[i] = float(base[models[i]]) / 7

    model_colors = plt.cm.Pastel1(model_labels)
    


    plt.figure(figsize=(6, 6))

    plt.axis('off')
    plt.axis('tight')

    # plt.table(cellText=data,
    #                       cellColours=cell_colors,
    #                       rowLabels=models,
    #                       rowColours=model_colors,
    #                       colLabels=models,
    #                       colColours=model_colors,
    #                       loc='center')
    
    plt.table(cellColours=cell_colors,
              rowLabels=models,
              rowColours=model_colors,
              colLabels=models,
              colColours=model_colors,
              loc='center')

    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    return

if __name__ == "__main__":
    # dict_ft = get_dict_ft("/nlp/u/salzhu/model-tracing/config/llama_flat.yaml")

    # plot_statistic_scatter("/juice4/scr4/nlp/model-tracing/llama_models_runs/mc_base_wikitext/logs", 
    #                dict_ft, "test_statistic_plots/mc_base_wikitext")

    plot_statistic_grid("/juice4/scr4/nlp/model-tracing/llama_models_runs/cos_acts_rand/logs", 
                        base_ordered, "test_statistic_tables/cos_acts_rand", 3, log=False)


