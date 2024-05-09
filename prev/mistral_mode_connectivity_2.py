import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
# ppl = torch.exp(loss)
# print(ppl)

model_ft = AutoModelForCausalLM.from_pretrained("alvarobartt/Mistral-7B-v0.1-ORPO")
tokenizer_ft = AutoTokenizer.from_pretrained("alvarobartt/Mistral-7B-v0.1-ORPO")

model_temp = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer_temp = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

target = dict(model.named_parameters())
target_ft = dict(model_ft.named_parameters())
target_temp = dict(model_temp.named_parameters())

print(
    "------------------------------------------------------------------------------------"
)

# print(target['model.layers.0.self_attn.q_proj.weight'].shape)
# print(target['model.layers.0.self_attn.k_proj.weight'].shape)
# print(target['model.layers.0.self_attn.v_proj.weight'].shape)
# print(target['model.layers.0.self_attn.o_proj.weight'].shape)
# print(target['model.layers.0.mlp.gate_proj.weight'].shape)
# print(target['model.layers.0.mlp.up_proj.weight'].shape)
# print(target['model.layers.0.mlp.down_proj.weight'].shape)
# print(target['model.layers.0.input_layernorm.weight'].shape)
# print(target['model.layers.0.post_attention_layernorm.weight'].shape)

# print("------------------------------------------------------------------------------------")

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

steps = np.arange(1, step=0.1)

losses = []

for step in steps:

    print(
        "------------------------------------------------------------------------------------"
    )

    with torch.no_grad():

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

        inputs = tokenizer_temp(
            "ABC is a startup based in New York City and Paris", return_tensors="pt"
        )
        loss = model_temp(
            input_ids=inputs["input_ids"], labels=inputs["input_ids"]
        ).loss
        ppl = torch.exp(loss)
        print(ppl)

        losses.append(ppl)


print(losses)
