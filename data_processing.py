import torch

from datasets import load_dataset, Dataset
from accelerate.data_loader import DataLoaderShard
import transformers
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
        break

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

def generate_texts(model_name, tokenizer, num, max_tokens=20, device="cuda"):
  pipeline = transformers.pipeline(
      "text-generation",
      model=model_name,
      torch_dtype=torch.float16,
      device_map=device
  )

  texts = []

  start = ""
  for i in range(num):
    generation = pipeline(start, max_new_tokens = max_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    texts.append(generation[0]['generated_text'])

  pipeline.model.to("cpu")
  del pipeline.model, pipeline
  torch.cuda.empty_cache()

  return texts

def evaluate_texts(model, tokenizer, texts, device="cuda"):
  model.to(device)
  losses = []
  for text in texts:
    inputs = tokenizer(text, return_tensors = "pt").to('cuda')
    temp = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
    losses.append(temp.item())
    inputs = inputs.to('cpu')
    del inputs

  model.to("cpu")
  return losses
