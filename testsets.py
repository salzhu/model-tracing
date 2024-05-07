import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import math
import numpy as np
import copy

block_size = 256

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def load_filtered_dataset(dataset, split, tokenizer):

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    eval_dataset = load_dataset(dataset, split=split)
    eval_dataset = eval_dataset.filter(lambda example: len(example["text"]) < 15)
    tokenized_datasets = eval_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, batch_size=1000, num_proc=4)
    return lm_datasets

