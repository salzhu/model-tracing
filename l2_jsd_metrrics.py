import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
import copy
import csv
import os
import os
import time

from l2_norm import calculate_l2_distance
from datasets import load_dataset
from permutation_tests import group_texts

import warnings
warnings.filterwarnings("ignore")

# os.environ['HF_HOME'] = '/nlp/scr/salzhu/hf'
# os.environ['PIP_CACHE_DIR'] = "/scr/salzhu/"
# os.environ['WANDB_CACHE_DIR'] = "/scr/salzhu/"

def compute_jsd_tokenized(model_a, model_b, input_ids, attention_mask, labels):
    with torch.no_grad():
        print("a")
        outputs_a = model_a(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        print("b")
        print(outputs_a)
        outputs_b = model_b(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        print("c")
        print(outputs_b)
        logits_a = outputs_a.logits.squeeze()
        logits_b = outputs_b.logits.squeeze()
        print("d")

        log_probs_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
        log_probs_b = torch.nn.functional.log_softmax(logits_b, dim=-1)
        print("e")

        log_probs_a = log_probs_a[:32000]
        log_probs_b = log_probs_b[:32000]

        print(log_probs_a.shape)
        print(log_probs_b.shape)

        m = 0.5 * (log_probs_a + log_probs_b)
        log_m = torch.logsumexp(m, dim=-1, keepdim=True)

        kl_div_a_m = (log_probs_a - log_m).sum(dim=-1)
        kl_div_b_m = (log_probs_b - log_m).sum(dim=-1)

        js_divergence = 0.5 * (kl_div_a_m + kl_div_b_m)
        print("f")

    return js_divergence.mean().item()

filepath = "/nlp/u/salzhu/model-tracing/model_pairs_jsd.csv"

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
        if(i == 0 and j <= 6): continue 
        print("Starting...")
        time0 = time.time()

        model_a_name = model_list[i]
        model_b_name = model_list[j]

        print(model_a_name, model_b_name)

        model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
        model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

        # unperm_l2, a, b = calculate_l2_distance(model_a, model_b)

        tokenizer = AutoTokenizer.from_pretrained(model_a_name)
        eval_dataset = load_dataset("dlwh/wikitext_103_detokenized", split="test")
        tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = eval_dataset.map(
            tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=8,
        )
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=1,
            do_eval=True,
            report_to=None,
            dataloader_num_workers=8,
            use_cpu=True,
        )
        trainer = Trainer(model=model_a, args=training_args, eval_dataset=lm_datasets)
        eval_dataloader = trainer.get_test_dataloader(lm_datasets)

        for batch in eval_dataloader:
            print(batch.keys())

            input_ids = batch["input_ids"]
            print(len(input_ids))
            print(input_ids)
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            jsd = compute_jsd_tokenized(model_a, model_b, input_ids, attention_mask, labels)
            break

        csv_header = ["Model Pair", "l2", "jsd"]

        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_header)
        
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            model_pair = f"{model_a_name} vs {model_b_name}"
            row = [model_pair, jsd]
            writer.writerow(row)

        csv_header = ["Model Pair", "l2"]
        
        print("done! time was")
        print(str(time.time() - time0))
