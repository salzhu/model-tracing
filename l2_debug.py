import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import torch.nn.functional as F
import torch.linalg as LA

import numpy as np

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
).to("cuda")

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


model_name = "EleutherAI/llemma_7b"
print(model_name, flush=True)

model_ft = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to("cuda")
target_ft = dict(model_ft.named_parameters())

l2_norms = []

for i in range(32):
    for layer in layer_2d_names:
        # import ipdb; ipdb.set_trace()
        layer_name = "model.layers." + str(i) + layer
        target_tensor = target[layer_name][:][:]
        target_ft_tensor = target_ft[layer_name][:][:]
        layer_l2 = torch.flatten(target_tensor) - torch.flatten(target_ft_tensor)

        loss_formal = torch.sum((target_tensor - target_ft_tensor) ** 2) ** 0.5
        loss_frob = LA.matrix_norm(target_tensor - target_ft_tensor)
        # so using np.dot on a flattened array does an inner product
        # and then you do an sqrt on that which is what we want
        l2_dot = np.dot(
            layer_l2.detach().cpu().float().numpy(),
            layer_l2.detach().cpu().float().numpy(),
        )
        layer_l2 = np.sqrt(l2_dot)

        l2_norms.append(layer_l2)
        diff = loss_formal.item() - layer_l2
        if diff > 1 or diff < -1:
            import ipdb

            ipdb.set_trace()
        print(f"layer {layer_name} {layer_l2}", flush=True)
        print(f" frob diff {loss_frob - loss_formal} formal", flush=True)
    for layer in layer_1d_names:
        layer_name = "model.layers." + str(i) + layer
        layer_l2 = torch.flatten(target[layer_name][:]) - torch.flatten(
            target_ft[layer_name][:]
        )
        # np.sqrt takes square root elementwise... we should sum all of them then take square root
        layer_l2_dot = np.dot(
            layer_l2.detach().cpu().float().numpy(),
            layer_l2.detach().cpu().float().numpy(),
        )
        layer_l2 = np.sqrt(layer_l2_dot)
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

# print(l2_norms, flush=True)
print(np.average(l2_norms), flush=True)
