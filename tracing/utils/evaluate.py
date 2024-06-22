import torch
import os
import glob
from typing import List, Optional
from datasets import load_dataset
from accelerate.data_loader import DataLoaderShard
from transformers import AutoTokenizer

# Llama2 context length
BLOCK_SIZE = 2048

def prepare_hf_dataset(hf_path,block_size,tokenizer,split="test"):
  raw_dataset = load_dataset(hf_path, split=split)
  dataset = raw_dataset.map(
        lambda examples : tokenize_function(examples,tokenizer), batched=True, remove_columns=["text"]
    ).map(
        lambda examples : group_texts(examples,block_size), batched=True, batch_size=1
    )
  dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
  return dataset


def prepare_programming_dataset(json_path: str, tokenizer: AutoTokenizer, columns_ignored: List[str]):
    columns_ignored = ['text', 'added', 'id', 'lang', 'metadata', 'source', 'timestamp', 'subdomain']
    raw_dataset = load_dataset("json", data_files=json_path)

    tokenized_datasets = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=columns_ignored
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1,
        num_proc=1,
    )
    return lm_datasets

def load_dolma_programming_datasets(
    test_name: str, 
    tokenizer: AutoTokenizer,
    columns_ignored: List[str],
):
    base_path = "/juice4/scr4/nlp/model-tracing/dolma_program_languages"
    
    
    json_dir = f"{base_path}/json_files_{test_name}"
    json_files = glob.glob(f"{json_dir}/*.json")
    
    datasets = {}
    for json_file in json_files:
        lang = os.path.splitext(os.path.basename(json_file))[0]
        datasets[lang] = prepare_programming_dataset(json_file, tokenizer, columns_ignored)
    
    return datasets

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

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result