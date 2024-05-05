import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import itertools
import os
from datasets import load_dataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def interpolate_models(model_a, model_b, alpha):
    # Implement the actual interpolation logic here
    return model_a

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

def main():
    # Automatically detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ["WANDB_WATCH"] = "false"



    # Load models and tokenizer
    model_list = [
        "meta-llama/Llama-2-7b-hf",
        "codellama/CodeLlama-7b-hf",
        "openlm-research/open_llama_7b",
    ]
    models = [AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) for model_name in model_list]
    tokenizer = AutoTokenizer.from_pretrained(models[0].config._name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    eval_dataset = load_dataset("dlwh/wikitext_103_detokenized", split="test")
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = eval_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
    )
    # eval_dataset = eval_dataset.filter(lambda example: len(example["text"]) < 30)
    # tokenized_dataset = eval_dataset.map(lambda example: tokenizer(example['text'], truncation=True, max_length=512), batched=True)
    # tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    # Prepare for evaluation
    training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=16, do_eval=True)
    model_pairs = list(itertools.combinations(enumerate(models), 2))

    # Evaluate perplexity across model interpolations
    for (idx_a, model_a), (idx_b, model_b) in tqdm(model_pairs, desc="Model Interpolation"):
        
        if idx_a < idx_b:
            perplexities = []
            for alpha in tqdm([round(alpha * 0.1, 2) for alpha in range(11)], desc="Alpha Perplexities"):
                interpolated_model = interpolate_models(model_a, model_b, alpha).to(device)
                trainer = Trainer(model=interpolated_model, args=training_args, eval_dataset=lm_datasets)
                result = trainer.evaluate()
                loss = result["eval_loss"]
                perplexity = math.exp(loss)
                perplexities.append(perplexity)
                interpolated_model.to("cpu")
                torch.cuda.empty_cache()

            # Output results
            print(f"Perplexities for interpolation between model {idx_a} and model {idx_b}: {perplexities}")

if __name__ == "__main__":
    main()
