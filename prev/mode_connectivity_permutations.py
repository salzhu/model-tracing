import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers

import numpy as np

# Loading fine tuned model to permute

model_ft = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5", torch_dtype=torch.bfloat16
)
tokenizer_ft = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
target_ft = dict(model_ft.named_parameters())

print(
    "------------------------------------------------------------------------------------",
    flush=True,
)

# Loading wikitext dataset

wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

test_dataset = wikitext_dataset["test"]

test_dataloader = DataLoader(test_dataset["text"], batch_size=1)

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

# all_losses = []

print(
    "------------------------------------------------------------------------------------",
    flush=True,
)

# permuting the layers of fine-tuned model

permutation = np.random.permutation(np.arange(4096))
permutation = torch.randperm(4096)

target_ft["model.embed_tokens.weight"] = target_ft["model.embed_tokens.weight"][
    :, permutation
]

for i in range(32):
    target_ft["model.layers." + str(i) + ".self_attn.q_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".self_attn.q_proj.weight"
    ][:, permutation]
    target_ft["model.layers." + str(i) + ".self_attn.k_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".self_attn.k_proj.weight"
    ][
        permutation
    ]  #
    target_ft["model.layers." + str(i) + ".self_attn.v_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".self_attn.v_proj.weight"
    ][:, permutation]
    target_ft["model.layers." + str(i) + ".self_attn.o_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".self_attn.o_proj.weight"
    ][
        permutation
    ]  #
    target_ft["model.layers." + str(i) + ".mlp.gate_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".mlp.gate_proj.weight"
    ][
        :, permutation
    ]  # fixed
    target_ft["model.layers." + str(i) + ".mlp.up_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".mlp.up_proj.weight"
    ][
        :, permutation
    ]  # fixed
    target_ft["model.layers." + str(i) + ".mlp.down_proj.weight"] = target_ft[
        "model.layers." + str(i) + ".mlp.down_proj.weight"
    ][
        permutation
    ]  # fixed
    target_ft["model.layers." + str(i) + ".input_layernorm.weight"] = target_ft[
        "model.layers." + str(i) + ".input_layernorm.weight"
    ][
        permutation
    ]  # 1d
    target_ft[
        "model.layers." + str(i) + ".post_attention_layernorm.weight"
    ] = target_ft["model.layers." + str(i) + ".post_attention_layernorm.weight"][
        permutation
    ]  # 1d

target_ft["model.norm.weight"] = target_ft["model.norm.weight"][permutation]
target_ft["lm_head.weight"] = target_ft["lm_head.weight"][:, permutation]

print("embed_tokens" + str(target_ft["model.embed_tokens.weight"].shape))
print(
    "self_attn.q_proj " + str(target_ft["model.layers.0.self_attn.q_proj.weight"].shape)
)
print(
    "self_attn.k_proj" + str(target_ft["model.layers.0.self_attn.k_proj.weight"].shape)
)
print(
    "self_attn.v_proj" + str(target_ft["model.layers.0.self_attn.v_proj.weight"].shape)
)
print(
    "self_attn.o_proj" + str(target_ft["model.layers.0.self_attn.o_proj.weight"].shape)
)
print("mlp.gate_proj" + str(target_ft["model.layers.0.mlp.gate_proj.weight"].shape))
print("mlp.up_proj" + str(target_ft["model.layers.0.mlp.up_proj.weight"].shape))
print("mlp.down_proj" + str(target_ft["model.layers.0.mlp.down_proj.weight"].shape))
print("input_layernorm" + str(target_ft["model.layers.0.input_layernorm.weight"].shape))
print(
    "post_attention_layernorm"
    + str(target_ft["model.layers.0.post_attention_layernorm.weight"].shape)
)
print("norm" + str(target_ft["model.norm.weight"].shape))
print("lm_head" + str(target_ft["lm_head.weight"].shape))

print()
print()

# Loading Llama2 models

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model_temp = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
)
tokenizer_temp = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

target = dict(model.named_parameters())
target_temp = dict(model_temp.named_parameters())

print("embed_tokens" + str(target["model.embed_tokens.weight"].shape))
print("self_attn.q_proj " + str(target["model.layers.0.self_attn.q_proj.weight"].shape))
print("self_attn.k_proj" + str(target["model.layers.0.self_attn.k_proj.weight"].shape))
print("self_attn.v_proj" + str(target["model.layers.0.self_attn.v_proj.weight"].shape))
print("self_attn.o_proj" + str(target["model.layers.0.self_attn.o_proj.weight"].shape))
print("mlp.gate_proj" + str(target["model.layers.0.mlp.gate_proj.weight"].shape))
print("mlp.up_proj" + str(target["model.layers.0.mlp.up_proj.weight"].shape))
print("mlp.down_proj" + str(target["model.layers.0.mlp.down_proj.weight"].shape))
print("input_layernorm" + str(target["model.layers.0.input_layernorm.weight"].shape))
print(
    "post_attention_layernorm"
    + str(target["model.layers.0.post_attention_layernorm.weight"].shape)
)
print("norm" + str(target["model.norm.weight"].shape))
print("lm_head" + str(target["lm_head.weight"].shape))

# Mode connectivity tests

steps = np.arange(1.1, step=0.1)

losses = []

for step in steps:

    print(step, end=" ", flush=True)

    with torch.no_grad():

        target_temp["model.embed_tokens.weight"][:][:] = (
            step * target["model.embed_tokens.weight"][:][:]
            + (1 - step) * target_ft["model.embed_tokens.weight"][:][:]
        )

        for i in range(32):
            for layer in layer_2d_names:
                layer_name = "model.layers." + str(i) + layer
                target_temp[layer_name][:][:] = (
                    step * target[layer_name][:][:]
                    + (1 - step) * target_ft[layer_name][:][:]
                )

            for layer in layer_1d_names:
                layer_name = "model.layers." + str(i) + layer
                target_temp[layer_name][:] = (
                    step * target[layer_name][:] + (1 - step) * target_ft[layer_name][:]
                )

        target_temp["model.norm.weight"][:] = (
            step * target["model.norm.weight"][:]
            + (1 - step) * target_ft["model.norm.weight"][:]
        )
        target_temp["lm_head.weight"][:][:] = (
            step * target["lm_head.weight"][:][:]
            + (1 - step) * target_ft["lm_head.weight"][:][:]
        )

        model_temp = model_temp.to("cuda")

        local_losses = []

        for item in test_dataloader:

            text = item[0]
            if text == "" or text[1] == "=":
                continue

            inputs = tokenizer_temp(text, return_tensors="pt").to("cuda")
            loss = model_temp(
                input_ids=inputs["input_ids"], labels=inputs["input_ids"]
            ).loss
            local_losses.append(loss.item())

        print(sum(local_losses) / len(local_losses))
        losses.append(sum(local_losses) / len(local_losses))

print(losses, flush=True)

print(
    "------------------------------------------------------------------------------------",
    flush=True,
)
