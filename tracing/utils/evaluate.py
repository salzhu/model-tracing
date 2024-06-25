import torch
import os
import glob
from typing import List, Optional
from datasets import load_dataset, concatenate_datasets
from accelerate.data_loader import DataLoaderShard
from transformers import AutoTokenizer



def prepare_hf_dataset(hf_path,block_size,tokenizer,split="test"):
  raw_dataset = load_dataset(hf_path, split=split)
  dataset = raw_dataset.map(
        lambda examples : tokenize_function(examples,tokenizer), batched=True, remove_columns=["text"]
    ).map(
        lambda examples : group_texts(examples,block_size), batched=True, batch_size=1
    )
  dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
  return dataset


def prepare_programming_dataset(json_path: str, block_size: int, tokenizer: AutoTokenizer, columns_ignored: List[str]):
    raw_dataset = load_dataset("json", data_files=json_path)

    dataset = raw_dataset["train"].map(
        lambda examples: tokenize_function(examples, tokenizer), 
        batched=True, 
        num_proc=4, 
        remove_columns=columns_ignored
    ).map(
        lambda examples: group_texts(examples, block_size), 
        batched=True, 
        batch_size=1,
        num_proc=1
    )
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def load_m2d2_datasets(
    test_name: str,
    block_size: int,
    tokenizer: AutoTokenizer,
    columns_ignored: List[str],
):
    base_path = "/juice4/scr4/nlp/model-tracing/m2d2_s2orc"
    json_dir = f"{base_path}/{test_name}"
    json_files = glob.glob(f"{json_dir}/*.json")
    
    if not json_files:
        raise ValueError(f"No JSON files found for test case: {test_name}")
    
    datasets = []
    for json_file in json_files:
        dataset = prepare_programming_dataset(json_file, block_size, tokenizer, columns_ignored)
        datasets.append(dataset)
    
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset

def load_dolma_programming_datasets(
    test_name: str, 
    block_size: int,
    tokenizer: AutoTokenizer,
    columns_ignored: List[str],
):
    base_path = "/juice4/scr4/nlp/model-tracing/dolma_program_languages"
    
    
    json_dir = f"{base_path}/json_files_{test_name}"
    json_files = glob.glob(f"{json_dir}/*.json")
    
    datasets = []
    for json_file in json_files:
        dataset = prepare_programming_dataset(json_file, block_size, tokenizer, columns_ignored)
        datasets.append(dataset)
    
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset

def prepare_hf_dataloader(dataset, batch_size: int):
    return DataLoaderShard(dataset, batch_size=batch_size)

def evaluate(model, dataloader, device: str = "cuda"):
    losses = []
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            losses.append(loss.item())

    model.to("cpu")
    return losses

def prepare_aya_dataset(subset: str, language: str, block_size: int, tokenizer: AutoTokenizer):
    """
    Prepare the Aya dataset for a specific subset and language.
    """
    raw_dataset = load_dataset("CohereForAI/aya_evaluation_suite", subset)
    filtered_dataset = raw_dataset.filter(lambda example: example['language'] == language)
    
    dataset = filtered_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=filtered_dataset.column_names
    ).map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        batch_size=1
    )
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def tokenize_aya_function(examples, tokenizer: AutoTokenizer):
    """
    Tokenize Aya dataset examples.
    """
    return tokenizer(examples['inputs'])

def tokenize_function(examples, tokenizer):
    if 'text' in examples:
        return tokenizer(examples['text'])
    elif 'inputs' in examples:
        return tokenizer(examples['inputs'])
    else:
        raise ValueError("Neither 'text' nor 'inputs' found in examples")

def group_texts(examples,block_size):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])

    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result