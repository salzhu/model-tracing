import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import itertools
import os
from datasets import load_dataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import csv
from utils import calculate_l2_distance, interpolate_models

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
    os.environ["WANDB_MODE"] = "disabled"

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
    eval_dataset = eval_dataset.filter(lambda example: len(example["text"]) < 15)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = eval_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, batch_size=1000, num_proc=4,
    )

    # Calculate the L2 distance between each pair of models
    model_pairs = list(itertools.combinations(enumerate(models), 2))
    l2_distances = {}
    for (idx_a, model_a), (idx_b, model_b) in tqdm(model_pairs, desc="Calculating L2 Distances"):
        if idx_a < idx_b:
            l2_distance = calculate_l2_distance(model_a, model_b)
            print(f"L2 distance between {model_a.config._name_or_path} and {model_b.config._name_or_path}: {l2_distance}")
            model_a_name = model_a.config._name_or_path.split("/")[-1]
            model_b_name = model_b.config._name_or_path.split("/")[-1]
            l2_distances[(model_a_name, model_b_name)] = l2_distance

    # Prepare for evaluation
    training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=16, do_eval=True, report_to=None)
    alphas = [round(alpha * 0.1, 2) for alpha in range(11)]

    # Evaluate perplexity across model interpolations
    for (idx_a, model_a), (idx_b, model_b) in tqdm(model_pairs, desc="Model Interpolation"):
        if idx_a < idx_b:
            perplexities = []
            model_a_name = model_a.config._name_or_path.split("/")[-1]
            model_b_name = model_b.config._name_or_path.split("/")[-1]
            for alpha in tqdm(alphas, desc=f"Alpha Perplexities for {model_a_name} and {model_b_name}"):
                interpolated_model = interpolate_models(model_a, model_b, alpha).to(device)
                trainer = Trainer(model=interpolated_model, args=training_args, eval_dataset=lm_datasets)
                result = trainer.evaluate()
                loss = result["eval_loss"]
                perplexity = math.exp(loss)
                perplexities.append(perplexity)
                interpolated_model.to("cpu")
                torch.cuda.empty_cache()

            # Save perplexities and model names to CSV
            csv_filename = "perplexities.csv"
            csv_header = ["Model Pair", "L2 Distance"] + [f"Alpha {alpha}" for alpha in alphas]
            if not os.path.exists(csv_filename):
                with open(csv_filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)

            with open(csv_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                model_pair = f"{model_a_name} vs {model_b_name}"
                l2_distance = l2_distances[(model_a_name, model_b_name)]
                row = [model_pair, l2_distance] + perplexities
                writer.writerow(row)

            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.plot(alphas, perplexities)
            plt.xlabel("Alpha")
            plt.ylabel("Perplexity")
            plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")

            # Save the plot as a PNG file
            plot_filename = f"alpha_vs_perplexity_{model_a_name}_vs_{model_b_name}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()