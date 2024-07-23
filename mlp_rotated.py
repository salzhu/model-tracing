import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ortho_group
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from collections import defaultdict

from tracing.utils.evaluate import evaluate, evaluate_csh_test
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets, prepare_random_sample_dataset
from tracing.statistics.cos_weights import statistic as csw
from tracing.statistics.csh_robust import act_robust
from tracing.statistics.csh_robust import act_rand_mlp as cshr
from tracing.statistics.l2 import calculate_l2_distance
from tracing.utils.utils import cossim
from tracing.statistics.cos import statistic as cswr
from tracing.statistics.jsd import statistic as jsd

def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def mlp_cshr_per_layer(base_model,ft_model,i,n=500,emb_size=16,mlp_dim=20):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    linear_combinations_gate = torch.rand(size=(n,1,mlp_dim), dtype=torch.float32)
    
    linear_combinations_up = torch.randn(size=(n,1,mlp_dim), dtype=torch.float32)
    x_base = torch.matmul(linear_combinations_gate, base_model.model.layers[0].mlp.gate_proj.weight)
    x_ft = torch.matmul(linear_combinations_gate, ft_model.model.layers[0].mlp.gate_proj.weight)

    # x_base += torch.matmul(linear_combinations_up, base_model.model.layers[i].mlp.up_proj.weight)
    # x_ft += torch.matmul(linear_combinations_up, ft_model.model.layers[i].mlp.up_proj.weight)

    x_base = torch.squeeze(x_base)
    x_ft = torch.squeeze(x_ft)

    # print(x_base.shape)

    with torch.no_grad():
        # base_model.to("cuda")
        y_base = base_model.model.layers[i].mlp(x_base)
        # base_model.to("cpu")
        
        # ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(x_ft)
        # ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    print(base_mat)
    print(ft_mat)
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    print(cossim(base_mat,ft_mat))
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def mlp_cshr_bad(base_model,ft_model,i,n=5000,emb_size=16,mlp_dim=20):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,emb_size), dtype=torch.float32)
    # print(x.shape)
    with torch.no_grad():
        y_base = base_model.model.layers[i].mlp(x)
        
        y_ft = ft_model.model.layers[i].mlp(x)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def mlp_cshr_full_model(base_model, ft_model, dataloader, num_layers=32):

    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"ft_"+layer)
        ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hooks[layer])

    evaluate_csh_test(base_model,dataloader)
    evaluate_csh_test(ft_model,dataloader)

    stats = []

    for i in range(num_layers):

        base_mat = torch.vstack(feats[f'base_{i}'])
        base_mat = base_mat.view(-1,base_mat.shape[-1]).T

        ft_mat = torch.vstack(feats[f'ft_{i}'])
        ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

        val = torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()
    
        stats.append(val)

    return stats

def main():

    model_name = "meta-llama/Llama-2-7b-hf"
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # model_rotated = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    hidden_dim = 16
    mlp_dim = 20
    vocab_size = 100
    num_heads = 4
    seq_length = 30
    batch_size = 2
    num_layers = 4

    config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden_dim, intermediate_size=mlp_dim, 
                        num_hidden_layers = num_layers, num_attention_heads=num_heads, 
                        max_position_embeddings=seq_length)

    model = LlamaForCausalLM(config)
    model_rotated = LlamaForCausalLM(config)

    weights = model.state_dict()
    weights_rotated = model_rotated.state_dict()

    for i in range(num_layers):
        weights[f'model.layers.{i}.input_layernorm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)
        weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)
    weights['model.norm.weight'] = torch.rand(hidden_dim, dtype=torch.float32)

    model.load_state_dict(weights)

    # print(evaluate(model, dataloader))

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = prepare_random_sample_dataset(5000, 512, vocab_size)
    # prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset,1)

    # cshr(model, model_rotated, num_layers=num_layers, emb_size=hidden_dim)
    # act_robust(model, model_rotated, dataloader, num_layers=num_layers)
    print("Random model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    # print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, num_layers)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with random inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=num_layers)))
    # print("mlp cshr with random canonical basis: " + str(mlp_cshr_per_layer(model, model_rotated, 0)))
    print()

    rotation = ortho_group.rvs(dim=hidden_dim)
    rotation = torch.tensor(rotation, dtype=torch.float32) # , dtype=torch.bfloat16

    weights_rotated['model.embed_tokens.weight'] = weights['model.embed_tokens.weight']@rotation

    for i in range(num_layers):
        weights_rotated[f'model.layers.{i}.input_layernorm.weight'] = torch.ones(hidden_dim)
        weights_rotated[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(hidden_dim)

        weights_rotated[f'model.layers.{i}.self_attn.q_proj.weight'] = weights[f'model.layers.{i}.self_attn.q_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
        weights_rotated[f'model.layers.{i}.self_attn.k_proj.weight'] = weights[f'model.layers.{i}.self_attn.k_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
        weights_rotated[f'model.layers.{i}.self_attn.v_proj.weight'] = weights[f'model.layers.{i}.self_attn.v_proj.weight']@torch.diag(weights[f'model.layers.{i}.input_layernorm.weight'])@rotation
        weights_rotated[f'model.layers.{i}.self_attn.o_proj.weight'] = rotation.T@weights[f'model.layers.{i}.self_attn.o_proj.weight'] 

        weights_rotated[f'model.layers.{i}.mlp.gate_proj.weight'] = weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])@rotation
        weights_rotated[f'model.layers.{i}.mlp.up_proj.weight'] = weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])@rotation
        weights_rotated[f'model.layers.{i}.mlp.down_proj.weight'] = rotation.T@weights[f'model.layers.{i}.mlp.down_proj.weight']

    weights_rotated['model.norm.weight'] = torch.ones(hidden_dim)
    weights_rotated['lm_head.weight'] = weights['lm_head.weight']@torch.diag(weights['model.norm.weight'])@rotation

    # model.load_state_dict(weights)
    model_rotated.load_state_dict(weights_rotated)

    x = torch.randint(low=0, high=vocab_size, size=(batch_size,seq_length))

    output = model(x).logits
    output_rotated = model_rotated(x).logits

    print("Rotated model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    # print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, num_layers)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with random inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=num_layers)))
    print("mlp cshr with random canonical basis: " + str(mlp_cshr_per_layer(model, model_rotated, 0)))
    

    # print(output)
    # print(output_rotated)

    # print("outputs are aligned: ")
    # print(abs(output - output_rotated) <= 0.0001)

    # print(evaluate(model, dataloader))

if __name__ == "__main__":
    main()



    # # linear_combinations_gate = torch.randn(size=(n,mlp_dim,1), dtype=torch.float32)
    # x_base = torch.matmul(base_model.model.layers[i].mlp.gate_proj.weight.T, linear_combinations_gate)
    # x_ft = torch.matmul(ft_model.model.layers[i].mlp.gate_proj.weight.T, linear_combinations_gate)