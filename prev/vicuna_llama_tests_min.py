import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

print("here!")

model_vicuna = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5", torch_dtype=torch.bfloat16
).to("cuda")
tokenizer_vicuna = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

model_llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
).to("cuda")
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

target_vicuna = dict(model_vicuna.named_parameters())

target_llama = dict(model_llama.named_parameters())

model_temp = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5", torch_dtype=torch.bfloat16
).to("cuda")
tokenizer_temp = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
target_temp = dict(model_temp.named_parameters())

print(
    "------------------------------------------------------------------------------------"
)

wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

test_dataset = wikitext_dataset["test"]

test_dataloader = DataLoader(test_dataset["text"], batch_size=1)

print(
    "------------------------------------------------------------------------------------"
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

steps = np.arange(1.1, step=0.1)

losses = []

for step in steps:

    print("***", flush=True)
    print(step, flush=True)

    with torch.no_grad():

        target_temp["model.embed_tokens.weight"][:][:] = (
            step * target_vicuna["model.embed_tokens.weight"][:][:]
            + (1 - step) * target_llama["model.embed_tokens.weight"][:32000][:]
        )

        for i in range(32):
            for layer in layer_2d_names:
                layer_name = "model.layers." + str(i) + layer
                target_temp[layer_name][:][:] = (
                    step * target_vicuna[layer_name][:][:]
                    + (1 - step) * target_llama[layer_name][:][:]
                )

            for layer in layer_1d_names:
                layer_name = "model.layers." + str(i) + layer
                target_temp[layer_name][:] = (
                    step * target_vicuna[layer_name][:]
                    + (1 - step) * target_llama[layer_name][:]
                )

        target_temp["model.norm.weight"][:] = (
            step * target_vicuna["model.norm.weight"][:]
            + (1 - step) * target_llama["model.norm.weight"][:]
        )
        target_temp["lm_head.weight"][:][:] = (
            step * target_vicuna["lm_head.weight"][:][:]
            + (1 - step) * target_llama["lm_head.weight"][:32000][:]
        )

        count = 0
        min_logits = 0

        for item in test_dataloader:

            text = item[0]
            if text == "" or text[1] == "=":
                continue

            count += 1
            inputs = tokenizer_temp(text, return_tensors="pt").to("cuda")

            logits = model_temp(
                input_ids=inputs["input_ids"], labels=inputs["input_ids"]
            ).logits
            preds = logits.softmax(dim=-1).tolist()
            probs_tensor = np.divide(preds[0][-1], sum(preds[0][-1]))

            min_logits += min(probs_tensor)

        min_logits /= count

        print(min_logits, flush=True)

        losses.append(min_logits)

print(losses)
