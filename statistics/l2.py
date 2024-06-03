# copied from Ahmed's l2_norm.py

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from torch.linalg import matrix_norm as LA_matrix_norm


def calculate_l2_norm(model):
    """
    Calculates the L2 norm for a single PyTorch model.

    Args:
        model (LlamaModel): The model to calculate the L2 norm for.

    Returns:
        tuple: A tuple containing the L2 norm, a list of tuples with layer names and their corresponding L2 norms, and the number of layers.
    """
    total_norm = 0
    layer_norms = []
    num_layers = 0

    running_norm = 0
    for name, param in model.named_parameters():
        if param.ndim != 2:
            param = param.reshape(-1, 1)
        if param.requires_grad:
            norm = LA_matrix_norm(param).item()
            total_norm += norm**2
            running_norm += norm
            layer_norms.append((name, norm))
            num_layers += 1

    total_norm = total_norm**0.5
    return total_norm, layer_norms, num_layers, running_norm


def calculate_l2_distance(model1, model2):
    """
    Calculates the L2 distance between two PyTorch models with the same architecture.
    Args:
        model1 (LlamaModel): The first model.
        model2 (LlamaModel): The second model.
    Returns:
        tuple: A tuple containing the L2 distance, a list of tuples with layer names and their corresponding L2 distances,
               and the number of layers.
    Raises:
        ValueError: If the parameter names or shapes do not match between the two models.
    """
    total_squared_diff = 0
    layer_distances = []
    num_layers = 0

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.shape != param2.shape:
            if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
                print(
                    f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )

        l2_diff = torch.sum((param1 - param2) ** 2) ** 0.5
        layer_l2_distance = l2_diff.item()
        total_squared_diff += layer_l2_distance
        layer_distances.append((name1, layer_l2_distance))
        num_layers += 1

    avg_l2_distance = total_squared_diff / num_layers
    return avg_l2_distance, layer_distances, num_layers


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


# Load the models
# model_a = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
# ).to("cuda")
# model_b = AutoModelForCausalLM.from_pretrained(
#     "EleutherAI/llemma_7b", torch_dtype=torch.bfloat16
# ).to("cuda")
# model_c = AutoModelForCausalLM.from_pretrained(
#     "codellama/CodeLlama-7b-hf", torch_dtype=torch.bfloat16
# ).to("cuda")
# # # Calculate the L2 distance between the two models and get the layer distances
# l2_norm_model_a, layer_norms_model_a, num_layers_a, running_norm_a = calculate_l2_norm(
#     model_a
# )
# print(f"L2 norm for model A: {l2_norm_model_a}")
# print(f"average l2 norm over layers: {l2_norm_model_a / num_layers_a}")
# print(f"average running norm over layers: {running_norm_a / num_layers_a}")
# l2_norm_model_b, layer_norms_model_b, num_layers_b, running_norm_b = calculate_l2_norm(
#     model_b
# )
# l2_norm_model_c, layer_norms_model_c, num_layers_c, running_norm_c = calculate_l2_norm(
#     model_c
# )
# print(f"L2 norm for model A: {l2_norm_model_a}")
# print(f"l2 norm for model B: {l2_norm_model_b}")
# print(f"average l2 norm over layers: {l2_norm_model_b / num_layers_b}")
# print(f"average running norm over layers: {running_norm_b / num_layers_b}")
# print(f"l2 norm for model C: {l2_norm_model_c}")
# print(f"average l2 norm over layers: {l2_norm_model_c / num_layers_c}")
# print(f"average running norm over layers: {running_norm_c / num_layers_c}")

# avg_l2_distance, layer_distances, num_layers = calculate_l2_distance(model_a, model_b)

# print(f"Average L2 distance per layer: {avg_l2_distance}")
# print(f"Number of layers: {num_layers}")
# print("Layer distances:")
# for layer_name, layer_distance in layer_distances:
#     print(f"Layer: {layer_name}, L2 distance: {layer_distance}")
