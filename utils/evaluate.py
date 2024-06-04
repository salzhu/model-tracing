import torch

from datasets import load_dataset
from accelerate.data_loader import DataLoaderShard

def prepare_hf_dataset(hf_path,block_size,tokenizer,split="test"):
  raw_dataset = load_dataset(hf_path, split=split)
  dataset = raw_dataset.map(
        lambda examples : tokenize_function(examples,tokenizer), batched=True, remove_columns=["text"]
    )
  dataset = dataset.map(
        lambda examples : group_texts(examples,block_size), batched=True, batch_size=1
    )
  dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

  return dataset

def prepare_hf_dataloader(dataset,batch_size):
  dataloader = DataLoaderShard(dataset,batch_size=batch_size)

  return dataloader

def evaluate(model,dataloader,device="cuda"):
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
        losses.append(loss)

  model.to("cpu")
  return losses
  
def tokenize_function(examples,tokenizer):
  return tokenizer(examples["text"])

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