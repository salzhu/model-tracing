import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from evaluate import load

import numpy as np

import pandas as pd

# model_vicuna = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.bfloat16).to('cuda')
# tokenizer_vicuna = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# model_codellama = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", torch_dtype=torch.bfloat16).to('cuda')
# tokenizer_codellama = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# model_llemma = AutoModelForCausalLM.from_pretrained("EleutherAI/llemma_7b", torch_dtype=torch.bfloat16).to('cuda')
# tokenizer_llemma = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")

# "lmsys/vicuna-7b-v1.5" "codellama/CodeLlama-7b-hf"

datasets = ["/nlp/scr/salzhu/vicuna_codellama_mean_prompts.csv", "/nlp/scr/salzhu/vicuna_llemma_mean_prompts.csv", "/nlp/scr/salzhu/codellama_llemma_mean_prompts.csv"]

for dataset in datasets:
    df = pd.read_csv(dataset)

    # count = 0
    # ppl = 0

    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=df['prompts'], model_id='lmsys/vicuna-7b-v1.5')

    print("Vicuna on " + dataset + ": " + str(results['mean_perplexity']))

    # for line in df['prompts']:
    #     inputs = tokenizer_vicuna(line, return_tensors = "pt").to('cuda')
    #     loss = model_vicuna(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
    #     ppl += torch.exp(loss).item()
    #     count += 1

    # ppl /= count
    # 

    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=df['prompts'], model_id="codellama/CodeLlama-7b-hf")

    print("Codellama on " + dataset + ": " + str(results['mean_perplexity']))
    

    # count = 0
    # ppl = 0

    # for line in df['prompts']:
    #     inputs = tokenizer_codellama(line, return_tensors = "pt").to('cuda')
    #     loss = model_codellama(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
    #     ppl += torch.exp(loss).item()
    #     count += 1

    # ppl /= count
    # print("Codellama on " + dataset + ": " + str(ppl))

    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=df['prompts'], model_id="EleutherAI/llemma_7b")

    print("Llemma on " + dataset + ": " + str(results['mean_perplexity']))

    # count = 0
    # ppl = 0

    # for line in df['prompts']:
    #     inputs = tokenizer_llemma(line, return_tensors = "pt").to('cuda')
    #     loss = model_llemma(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
    #     ppl += torch.exp(loss).item()
    #     count += 1

    # ppl /= count
    # print("Llemma on " + dataset + ": " + str(ppl))

