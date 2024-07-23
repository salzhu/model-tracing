import torch
import copy
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def permute_model(model, mlp_permutation, emb_permutation, n_blocks=32):
    permute_embedding_layer(model, emb_permutation)
    for i in range(n_blocks):
        permute_transformer_block(model, i, mlp_permutation, emb_permutation)
    permute_output_layer(model, emb_permutation)

def permute_transformer_block(model, i, mlp_permutation, emb_permutation):
    model.state_dict()['model.layers.' + str(i) + '.self_attn.q_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.self_attn.q_proj.weight'][:, emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.self_attn.k_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.self_attn.k_proj.weight'][:, emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.self_attn.v_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.self_attn.v_proj.weight'][:, emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.self_attn.o_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.self_attn.o_proj.weight'][emb_permutation]

    model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'][mlp_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'][mlp_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'][:, mlp_permutation]

    model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'][:, emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'][:, emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'][emb_permutation]

    model.state_dict()['model.layers.' + str(i) + '.input_layernorm.weight'] = model.state_dict()['model.layers.' + str(i) + '.input_layernorm.weight'][emb_permutation]
    model.state_dict()['model.layers.' + str(i) + '.post_attention_layernorm.weight'] = model.state_dict()['model.layers.' + str(i) + '.post_attention_layernorm.weight'][emb_permutation]

def permute_embedding_layer(model, emb_permutation):
    model.state_dict()['model.embed_tokens.weight'] = model.state_dict()['model.embed_tokens.weight'][:, emb_permutation]

def permute_output_layer(model, emb_permutation):
    model.state_dict()['lm_head.weight'] = model.state_dict()['lm_head.weight'][:, emb_permutation]
    model.state_dict()['model.norm.weight'] = model.state_dict()['model.norm.weight'][emb_permutation]

def permute_mlp_block(model, i, mlp_permutation):
    model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'][mlp_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'][mlp_permutation]
    model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'] = model.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'][:, mlp_permutation]

def avg_mlp_block(model0, model1, i, alpha=0.5):
    model0.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight']
    model0.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.mlp.up_proj.weight']
    model0.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.mlp.down_proj.weight']

def avg_transformer_block(model0, model1, i, alpha=0.5, attn=True):
    if attn:
        model0.state_dict()['model.layers.' + str(i) + '.self_attn.q_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.self_attn.q_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.self_attn.q_proj.weight']
        model0.state_dict()['model.layers.' + str(i) + '.self_attn.k_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.self_attn.k_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.self_attn.k_proj.weight']
        model0.state_dict()['model.layers.' + str(i) + '.self_attn.v_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.self_attn.v_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.self_attn.v_proj.weight']
        model0.state_dict()['model.layers.' + str(i) + '.self_attn.o_proj.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.self_attn.o_proj.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.self_attn.o_proj.weight']

    avg_mlp_block(model0, model1, i, alpha=alpha)
    
    model0.state_dict()['model.layers.' + str(i) + '.input_layernorm.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.input_layernorm.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.input_layernorm.weight']
    model0.state_dict()['model.layers.' + str(i) + '.post_attention_layernorm.weight'] = alpha * model0.state_dict()['model.layers.' + str(i) + '.post_attention_layernorm.weight'] + (1 - alpha) * model1.state_dict()['model.layers.' + str(i) + '.post_attention_layernorm.weight']

def avg_embedding_layer(model0, model1, alpha=0.5):
    model0.state_dict()['model.embed_tokens.weight'] = alpha * model0.state_dict()['model.embed_tokens.weight'] + (1 - alpha) * model1.state_dict()['model.embed_tokens.weight']

def avg_output_layer(model0, model1, alpha=0.5):
    model0.state_dict()['lm_head.weight'] = alpha * model0.state_dict()['lm_head.weight'] + (1 - alpha) * model1.state_dict()['lm_head.weight']
    model0.state_dict()['model.norm.weight'] = alpha * model0.state_dict()['model.norm.weight'] + (1 - alpha) * model1.state_dict()['model.norm.weight']

def avg_model(avg_model, model1, alpha=0.5, n_blocks=80, attn=True, emb=False):
    if emb:
        avg_embedding_layer(avg_model, model1, alpha=alpha)
    for i in range(n_blocks):
        avg_transformer_block(avg_model, model1, i, alpha=alpha, attn=attn)
    if emb:
        avg_output_layer(avg_model, model1, alpha=alpha)
    
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf', state_dict=avg_model.state_dict(), quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda:0")
    return model

def get_mlp_weights(model, i):
    return model.state_dict()['model.layers.' + str(i) + '.mlp.gate_proj.weight']

def get_emb_weights(model):
    return model.state_dict()['model.embed_tokens.weight']