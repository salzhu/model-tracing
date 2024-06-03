import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import math
import numpy as np
import copy
from tqdm import tqdm

from utils import interpolate_models
from testsets import load_filtered_dataset

block_size = 512

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

def unpermuted_mode_connectivity(model_base_name, model_ft_name, alpha_step=0.1, endpoints=True):

    # Load base model, i.e. Llama2
    model_base = AutoModelForCausalLM.from_pretrained(model_base_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_base_name)

    # Load fine tuned model to permute
    model_ft = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=torch.bfloat16)

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

    trainer = Trainer(model=model_ft, args=training_args, eval_dataset=lm_datasets)
    eval_dataloader = trainer.get_test_dataloader(lm_datasets)

    losses = []
    perplexities = []

    alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]
    if endpoints == False:
        alphas = alphas[1:-1]

    device = 'cuda'

    for alpha in alphas:
        interpolated_model = interpolate_models(model_base, model_ft, alpha).to('cuda')

        loss = []
        
        for batch in tqdm(eval_dataloader):
            # HF Trainer finds GPU by default
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = interpolated_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss.append(outputs.loss.item())

            input_ids = input_ids.to("cpu")
            attention_mask = attention_mask.to("cpu")
            labels = labels.to("cpu")
            torch.cuda.empty_cache()
            break

        loss_mean = sum(loss) / len(loss)
        perplexity = math.exp(loss_mean)

        perplexities.append(perplexity)
        losses.append(loss_mean)

        interpolated_model.to("cpu")
        torch.cuda.empty_cache()
        del interpolated_model, input_ids, attention_mask, labels, outputs, loss
        torch.cuda.empty_cache()

        print("alpha = " + str(alpha) + " | " + str(loss_mean) + " | " + str(perplexity))

    return losses, perplexities
