import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from tqdm import tqdm
import csv
from utils import calculate_l2_distance

def main():
    # Automatically detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models and tokenizer
    model_list = [
        "meta-llama/Llama-2-7b-hf",
        "codellama/CodeLlama-7b-hf",
        # Add more models as needed
    ]
    models = [
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for model_name in model_list
    ]
    tokenizer = AutoTokenizer.from_pretrained(models[0].config._name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Calculate the L2 distance between each pair of models
    model_pairs = list(itertools.combinations(enumerate(models), 2))
    l2_distances = {}
    for (idx_a, model_a), (idx_b, model_b) in tqdm(
        model_pairs, desc="Calculating L2 Distances"
    ):
        if idx_a < idx_b:
            l2_distance = calculate_l2_distance(model_a, model_b)
            print(
                f"L2 distance between {model_a.config._name_or_path} and {model_b.config._name_or_path}: {l2_distance}"
            )
            model_a_name = model_a.config._name_or_path.split("/")[-1]
            model_b_name = model_b.config._name_or_path.split("/")[-1]
            l2_distances[(model_a_name, model_b_name)] = l2_distance

    # Save L2 distances and model names to CSV
    csv_filename = "l2_distances.csv"
    csv_header = ["Model A", "Model B", "L2 Distance"]
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        for (model_a_name, model_b_name), distance in l2_distances.items():
            writer.writerow([model_a_name, model_b_name, distance])

if __name__ == "__main__":
    main()