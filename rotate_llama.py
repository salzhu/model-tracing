import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ortho_group
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

from tracing.utils.evaluate import evaluate, evaluate_csh_test
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets, prepare_random_sample_dataset
from tracing.statistics.cos_weights import statistic as csw
from tracing.statistics.csh_robust import act_robust
from tracing.statistics.csh_robust import act_rand_mlp as cshr
from tracing.statistics.l2 import calculate_l2_distance
from tracing.statistics.cos import statistic as cswr

torch.set_default_dtype(torch.bfloat16)

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
model_rotated = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')

hidden_dim = 16 # 4096
head_dim = 16 # 4096
mlp_dim = 20
vocab_size = 100
num_heads = 4
seq_length = 30
batch_size = 2
num_layers = 32 # 4

hidden_dim = 4096

# config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden_dim, intermediate_size=mlp_dim, 
#                      num_hidden_layers = num_layers, num_attention_heads=num_heads, 
#                      max_position_embeddings=seq_length)

# model = LlamaForCausalLM(config)
# model_rotated = LlamaForCausalLM(config)

weights = model.state_dict()
weights_rotated = model_rotated.state_dict()

# print(weights_rotated['model.layers.0.mlp.gate_proj.weight'].dtype)

# for i in range(num_layers):
#     weights[f'model.layers.{i}.input_layernorm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)
#     weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)
# weights['model.norm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)


base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer) 
# prepare_random_sample_dataset(100, 512, vocab_size)
# 
dataloader = prepare_hf_dataloader(dataset,1)

evaluate_orig = evaluate(model, dataloader)

print(evaluate_orig)

# cshr(model, model_rotated, num_layers=num_layers, emb_size=hidden_dim)
# act_robust(model, model_rotated, dataloader, num_layers=num_layers)

rotation = ortho_group.rvs(dim=hidden_dim)
rotation = torch.tensor(rotation, dtype=torch.bfloat16).to('cuda') # , dtype=torch.bfloat16

weights_rotated['model.embed_tokens.weight'] = weights['model.embed_tokens.weight']@rotation

for i in range(num_layers):
    # print(i)

    # weights[f'model.layers.{i}.input_layernorm.weight'] = torch.ones(hidden_dim)
    # weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(hidden_dim)

    weights_rotated[f'model.layers.{i}.input_layernorm.weight'] = torch.ones(hidden_dim)
    weights_rotated[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(hidden_dim)

    weights_rotated[f'model.layers.{i}.self_attn.q_proj.weight'] = weights[f'model.layers.{i}.self_attn.q_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
    weights_rotated[f'model.layers.{i}.self_attn.k_proj.weight'] = weights[f'model.layers.{i}.self_attn.k_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
    weights_rotated[f'model.layers.{i}.self_attn.v_proj.weight'] = weights[f'model.layers.{i}.self_attn.v_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
    weights_rotated[f'model.layers.{i}.self_attn.o_proj.weight'] = rotation.T@weights[f'model.layers.{i}.self_attn.o_proj.weight'] 

    weights_rotated[f'model.layers.{i}.mlp.gate_proj.weight'] = weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])@rotation
    weights_rotated[f'model.layers.{i}.mlp.up_proj.weight'] = weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])@rotation
    weights_rotated[f'model.layers.{i}.mlp.down_proj.weight'] = rotation.T@weights[f'model.layers.{i}.mlp.down_proj.weight']

# weights['model.norm.weight'] = torch.ones(hidden_dim)
weights_rotated['model.norm.weight'] = torch.ones(hidden_dim)
weights_rotated['lm_head.weight'] = weights['lm_head.weight']@torch.diag(weights['model.norm.weight'])@rotation

for i in range(num_layers):
    weights[f'model.layers.{i}.mlp.gate_proj.weight'] = weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
    weights[f'model.layers.{i}.mlp.up_proj.weight'] = weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
    weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(hidden_dim)

for i in range(num_layers):
    weights[f'model.layers.{i}.self_attn.q_proj.weight'] = weights[f'model.layers.{i}.self_attn.q_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])
    weights[f'model.layers.{i}.self_attn.k_proj.weight'] = weights[f'model.layers.{i}.self_attn.k_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])
    weights[f'model.layers.{i}.self_attn.v_proj.weight'] = weights[f'model.layers.{i}.self_attn.v_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])
    weights[f'model.layers.{i}.input_layernorm.weight'] = torch.ones(hidden_dim)

# model.load_state_dict(weights)
# model.to('cuda')
# weights = model.state_dict()


model.load_state_dict(weights)
model_rotated.load_state_dict(weights_rotated)

evaluate_rot = evaluate(model_rotated, dataloader)
print(evaluate_rot)

rotation.to('cpu')

# x = torch.randint(low=0, high=vocab_size, size=(batch_size,seq_length)).to('cuda')

# model.to('cuda')
# model_rotated.to('cuda')

# print("outputs")
# output = model(x).logits
# output_rotated = model_rotated(x).logits
# print(output)
# print(output_rotated)
# print()



# print("outputs are aligned: ")
# print(abs(output - output_rotated) <= 0.0001)

print("outputs are aligned: ")
print([abs(evaluate_orig[i] - evaluate_rot[i]) <= 0.01 for i in range(len(evaluate_orig))])

# calculate_l2_distance(model, model_rotated)
# csw(model, model_rotated)
# cshr(model, model_rotated, num_layers=num_layers, emb_size=hidden_dim)

# act_robust(model, model_rotated, dataloader, num_layers=num_layers)

model.to('cuda')

print("invariant 1")
print(weights['model.embed_tokens.weight']@weights['model.layers.0.self_attn.q_proj.weight'].T@weights['model.layers.0.self_attn.k_proj.weight']@weights['model.embed_tokens.weight'].T)
print("invariant 2")
print(weights['model.embed_tokens.weight']@weights['model.layers.0.self_attn.v_proj.weight'].T@weights['model.layers.0.self_attn.o_proj.weight'].T@weights['model.embed_tokens.weight'].T)
print("invariant 3")
print(weights['model.embed_tokens.weight']@weights[f'model.layers.{i}.mlp.up_proj.weight'].T@weights[f'model.layers.{i}.mlp.down_proj.weight'].T@weights['model.embed_tokens.weight'].T)
print()

model.to('cpu')
model_rotated.to('cuda')

print("rotated")
print("invariant 1")
print(weights_rotated['model.embed_tokens.weight']@weights_rotated['model.layers.0.self_attn.q_proj.weight'].T@weights_rotated['model.layers.0.self_attn.k_proj.weight']@weights_rotated['model.embed_tokens.weight'].T)
print("invariant 2")
print(weights_rotated['model.embed_tokens.weight']@weights_rotated['model.layers.0.self_attn.v_proj.weight'].T@weights_rotated['model.layers.0.self_attn.o_proj.weight'].T@weights_rotated['model.embed_tokens.weight'].T)
print("invariant 3")
print(weights_rotated['model.embed_tokens.weight']@weights_rotated[f'model.layers.{i}.mlp.up_proj.weight'].T@weights_rotated[f'model.layers.{i}.mlp.down_proj.weight'].T@weights_rotated['model.embed_tokens.weight'].T)
print()


# print(weights['model.embed_tokens.weight']@weights[f'model.layers.{i}.mlp.up_proj.weight']@weights[f'model.layers.{i}.mlp.down_proj.weight']@weights['model.embed_tokens.weight'].T)
# print(weights_rotated['model.embed_tokens.weight']@weights_rotated[f'model.layers.{i}.mlp.up_proj.weight']@weights_rotated[f'model.layers.{i}.mlp.down_proj.weight']@weights_rotated['model.embed_tokens.weight'].T)

# print(evaluate(model, dataloader))