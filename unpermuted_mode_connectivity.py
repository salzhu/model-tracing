import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import math
import numpy as np
import copy

from utils import interpolate_models, evaluate
from testsets import load_filtered_dataset

def unpermuted_mode_connectivity(model_base_name, model_ft_name, alpha_step=0.1):

    # Load base model, i.e. Llama2
    model_base = AutoModelForCausalLM.from_pretrained(model_base_name, torch_dtype=torch.bfloat16)
    tokenizer_base = AutoTokenizer.from_pretrained(model_base_name)

    # Load fine tuned model to permute
    model_ft = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=torch.bfloat16)

    testset = load_filtered_dataset("dlwh/wikitext_103_detokenized", "test", tokenizer_base)

    losses = []
    perplexities = []

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]

    for alpha in alphas:
        interpolated_model = interpolate_models(model_base, model_ft, alpha).to('cuda')
        loss, perplexity = evaluate(interpolated_model, testset)
        perplexities.append(perplexity)
        losses.append(loss)
        interpolated_model.to("cpu")
        torch.cuda.empty_cache()
        print("alpha = " + str(alpha) + " | " + str(loss) + " | " + str(perplexity))

    return losses, perplexities

base = "meta-llama/Llama-2-7b-hf"
base_short_name = "llama2-7b"

ft = "lmsys/vicuna-7b-v1.5"
ft_short_name = "vicuna-1.5-7b"
losses, perplexities = unpermuted_mode_connectivity(base, ft)

print("Losses: ", end = '')
print(losses)

print("Perplexities: ", end = '')
print(perplexities)
