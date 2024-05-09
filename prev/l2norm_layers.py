import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers

import numpy as np

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

target = dict(model.named_parameters())

print(
    "------------------------------------------------------------------------------------",
    flush=True,
)

layer_2d_names = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
]
layer_1d_names = [".input_layernorm.weight", ".post_attention_layernorm.weight"]

model_names_good = [
    "yahma/llama-7b-hf",
    "LLM360/Amber",
    "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    "oh-yeontaek/llama-2-7B-LoRA-assemble",
    "lvkaokao/llama2-7b-hf-instruction-lora",
    "lmsys/vicuna-7b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b",
    "codellama/CodeLlama-7b-hf",
    "EleutherAI/llemma_7b",
]

model_name = model_names_good[8]

print(model_name, flush=True)

model_ft = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to("cuda")
tokenizer_ft = AutoTokenizer.from_pretrained(model_name, use_fast=False)
target_ft = dict(model_ft.named_parameters())

l2_norms = []

for i in range(32):
    for layer in layer_2d_names:
        layer_name = "model.layers." + str(i) + layer
        layer_l2 = torch.flatten(target[layer_name][:][:]) - torch.flatten(
            target_ft[layer_name][:][:]
        )
        layer_l2 = np.sqrt(
            np.dot(
                layer_l2.detach().cpu().float().numpy(),
                layer_l2.detach().cpu().float().numpy(),
            )
        )
        l2_norms.append(layer_l2)
    for layer in layer_1d_names:
        layer_name = "model.layers." + str(i) + layer
        layer_l2 = torch.flatten(target[layer_name][:]) - torch.flatten(
            target_ft[layer_name][:]
        )
        layer_l2 = np.sqrt(
            np.dot(
                layer_l2.detach().cpu().float().numpy(),
                layer_l2.detach().cpu().float().numpy(),
            )
        )
        l2_norms.append(layer_l2)

layer_name = "model.norm.weight"
layer_l2 = torch.flatten(target[layer_name][:]) - torch.flatten(
    target_ft[layer_name][:]
)
layer_l2 = np.sqrt(
    np.dot(
        layer_l2.detach().cpu().float().numpy(), layer_l2.detach().cpu().float().numpy()
    )
)
l2_norms.append(layer_l2)

# layer_name = 'lm_head.weight'
# layer_l2 = torch.flatten(target[layer_name][:]) - torch.flatten(target_ft[layer_name][:])
# layer_l2 = np.sqrt(np.dot(layer_l2.detach().cpu().float().numpy(), layer_l2.detach().cpu().float().numpy()))
# l2_norms.append(layer_l2)

print(l2_norms, flush=True)
print(np.average(l2_norms), flush=True)
