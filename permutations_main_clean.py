import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import os
import time
import math
import datetime
import yaml
from yaml import load, Loader

from data_processing import prepare_hf_dataloader, prepare_hf_dataset, evaluate, generate_texts, evaluate_texts
from utils import interpolate_models
from l2_norm import calculate_l2_distance
from permute import permute_model

import warnings
warnings.filterwarnings("ignore")

def mode_connectivity_wikitext(model_a, model_b, alphas, device='cuda'):

    model_a.to('cpu')
    model_b.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_a.config._name_or_path)
    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,tokenizer,split="test")
    dataloader = prepare_hf_dataloader(dataset,100)

    losses = []
    perplexities = []

    for alpha in alphas:
        interpolated_model = interpolate_models(model_a, model_b, alpha).to(device)

        loss = evaluate(interpolated_model, dataloader)
        loss_mean = sum(loss) / len(loss)

        loss_mean = loss_mean.item()

        losses.append(loss_mean)
        perplexities.append(math.exp(loss_mean))

        interpolated_model.to("cpu")
        torch.cuda.empty_cache()
        del interpolated_model, loss
        torch.cuda.empty_cache()

        print("alpha = " + str(alpha) + " | " + str(loss_mean) + " | " + str(math.exp(loss_mean)))

    return losses, perplexities

def mode_connectivity_generated_text(model_a, model_b, texts_a, texts_b, tokenizer, alphas, device='cuda'):

    model_a.to('cpu')
    model_b.to('cpu')
    
    losses = []
    perplexities = []

    for alpha in alphas:
        interpolated_model = interpolate_models(model_a, model_b, alpha).to(device)

        loss_a = evaluate_texts(interpolated_model, tokenizer, texts_a)
        loss_b = evaluate_texts(interpolated_model, tokenizer, texts_b)

        loss_mean = (sum(loss_a) + sum(loss_b)) / (len(loss_a) + len(loss_b))

        losses.append(loss_mean)
        perplexities.append(math.exp(loss_mean))

        interpolated_model.to("cpu")
        torch.cuda.empty_cache()
        del interpolated_model
        torch.cuda.empty_cache()

        # print("alpha = " + str(alpha) + " | " + str(loss_mean) + " | " + str(math.exp(loss_mean)))

    return losses, perplexities

def permutation_tests_wikitext(model_a_name, model_b_name, num_perm, alpha=0.5, device="cuda"):

    torch.manual_seed(datetime.datetime.now().timestamp())

    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

    losses = []
    l2s = []
    norm_losses = []

    unperm_l2, _, _ = calculate_l2_distance(model_a, model_b)
    unperm_losses, _ = mode_connectivity_wikitext(model_a, model_b, alphas=[0, alpha, 1])
    unperm_loss_midpoint = unperm_losses[1]
    unperm_norm_loss_midpoint = unperm_loss_midpoint - (unperm_losses[0] + unperm_losses[2]) / 2
    print(unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, flush=True)

    for i in range(num_perm):
        print()
        print("Permutation " + str(i+1) + "/" + str(num_perm), end=" ", flush=True)

        mlp_permutation = torch.randperm(11008)
        emb_permutation = torch.randperm(4096)

        permute_model(model_b, mlp_permutation, emb_permutation)
        loss, = mode_connectivity_wikitext(model_a, model_b, [alpha], device)[0]
        l2 = calculate_l2_distance(model_a, model_b)[0]

        losses.append(loss)
        l2s.append(l2)
        norm_losses.append(loss - (unperm_losses[0] + unperm_losses[2]) / 2)
        print(l2, loss, loss - (unperm_losses[0] + unperm_losses[2]) / 2)

    return unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, l2s, losses, norm_losses

def permutation_tests_generated_text(model_a_name, model_b_name, num_perm, alpha=0.5, device="cuda"):

    torch.manual_seed(datetime.datetime.now().timestamp())

    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

    tokenizer_a = AutoTokenizer.from_pretrained(model_a.config._name_or_path)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b.config._name_or_path)

    texts_a = generate_texts(model_a.config._name_or_path, tokenizer_a, 10)
    texts_b = generate_texts(model_b.config._name_or_path, tokenizer_b, 10)

    losses = []
    l2s = []
    norm_losses = []

    unperm_l2, _, _ = calculate_l2_distance(model_a, model_b)
    unperm_losses, _ = mode_connectivity_generated_text(model_a, model_b, texts_a, texts_b, tokenizer_a, alphas=[0, alpha, 1], device=device)
    unperm_loss_midpoint = unperm_losses[1]
    unperm_norm_loss_midpoint = unperm_loss_midpoint - (unperm_losses[0] + unperm_losses[2]) / 2
    print(unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, flush=True)

    for i in range(num_perm):
        print()
        print("Permutation " + str(i+1) + "/" + str(num_perm), flush=True)

        mlp_permutation = torch.randperm(11008)
        emb_permutation = torch.randperm(4096)

        permute_model(model_b, mlp_permutation, emb_permutation)
        loss, = mode_connectivity_generated_text(model_a, model_b, texts_a, texts_b, tokenizer_a, alphas=[alpha], device=device)[0]
        l2 = calculate_l2_distance(model_a, model_b)[0]

        losses.append(loss)
        l2s.append(l2)
        norm_losses.append(loss - (unperm_losses[0] + unperm_losses[2]) / 2)

    return unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, l2s, losses, norm_losses

def p_value(unpermuted, permuted): 
    count = 0
    for a in permuted:
        if(a < unpermuted):
            count += 1

    return round((count + 1) / (len(permuted) + 1), 2)


def save_results_to_file(filename, model_a_name, model_b_name, p_value, unpermuted, permuted): 
    csv_header = ["model pair", "p-value", "unpermuted metric", "permuted metrics"]

    if not os.path.exists(filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        model_pair = f"{model_a_name} vs {model_b_name}"
        row = [model_pair, p_value, unpermuted] + permuted
        writer.writerow(row)


def main():

    stream = open("model_list.yaml", 'r')
    dictionary = yaml.load(stream, Loader=Loader)
    model_list = dictionary["model_list"]

    filename_l2_results = "/nlp/u/salzhu/permutation_l2_test_results.csv"
    filename_loss_results = "/nlp/u/salzhu/permutation_loss_test_results.csv"
    filename_norm_loss_results = "/nlp/u/salzhu/permutation_norm_loss_test_results.csv"

    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            print("Starting...")
            time0 = time.time()
            print(model_list[i], model_list[j])
            # unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, l2s, losses, norm_losses = permutation_tests_wikitext(model_list[i], model_list[j], 9)
            unperm_l2, unperm_loss_midpoint, unperm_norm_loss_midpoint, l2s, losses, norm_losses = permutation_tests_generated_text(model_list[i], model_list[j], 9)
            l2_p_value = p_value(unperm_l2, l2s)
            loss_p_value = p_value(unperm_loss_midpoint, losses)
            norm_loss_p_value = p_value(unperm_norm_loss_midpoint, norm_losses)

            print(l2_p_value, loss_p_value, norm_loss_p_value)

            save_results_to_file(filename_l2_results, model_list[i], model_list[j], l2_p_value, unperm_l2, l2s)
            save_results_to_file(filename_loss_results, model_list[i], model_list[j], loss_p_value, unperm_loss_midpoint, losses)
            save_results_to_file(filename_norm_loss_results, model_list[i], model_list[j], norm_loss_p_value, unperm_norm_loss_midpoint, norm_losses)

            print("Done! running time")
            print(str(time.time() - time0))


if __name__ == "__main__":
    main()
