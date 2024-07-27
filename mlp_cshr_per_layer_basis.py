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
from tracing.utils.llama.model import permute_model
from tracing.utils.llama.matching import match_wmats

torch.set_default_dtype(torch.bfloat16)

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def mlp_cshr_full_model(base_model, ft_model, dataloader, num_layers=32):

    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"ft_"+layer)
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

def mlp_cshr_per_layer(base_model,ft_model,i,n=5000,emb_size=4096,mlp_dim=11008):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    base_model.to("cuda")
    
    linear_combinations_gate = torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")
    linear_combinations_up = torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")

    x_base = torch.matmul(linear_combinations_gate, base_model.model.layers[i].mlp.gate_proj.weight) # * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight
    # x_base += torch.matmul(linear_combinations_up, base_model.model.layers[i].mlp.up_proj.weight) # * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight
    x_base = torch.squeeze(x_base)

    with torch.no_grad():
        y_base = base_model.model.layers[i].mlp(x_base)
        base_model.to("cpu")

    ft_model.to("cuda")
    
    x_ft = torch.matmul(linear_combinations_gate, ft_model.model.layers[i].mlp.gate_proj.weight) # * ft_model.model.layers[i].post_attention_layernorm.weight* ft_model.model.layers[i].post_attention_layernorm.weight
    # x_ft += torch.matmul(linear_combinations_up, ft_model.model.layers[i].mlp.up_proj.weight) # * ft_model.model.layers[i].post_attention_layernorm.weight * ft_model.model.layers[i].post_attention_layernorm.weight
    x_ft = torch.squeeze(x_ft)
    # print(x_ft.shape)

    # print(x_base, x_ft)

    with torch.no_grad():    
        # ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(x_ft)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    mat = cossim(base_mat,ft_mat).to('cpu')

    print(mat)
    
    base_handle.remove()
    ft_handle.remove()

    # print(torch.max(cossim(base_mat,ft_mat),axis=-1).values, torch.max(cossim(base_mat,ft_mat),axis=-1).indices)
    
    return torch.median(torch.max(mat,axis=-1).values).item()

def rotate_model(model, num_layers=32, hidden_dim=4096):

    model.to('cuda')

    rotation = ortho_group.rvs(dim=hidden_dim)
    rotation = torch.tensor(rotation, dtype=torch.bfloat16).to('cuda')

    weights = model.state_dict()
    weights_rotated = model.state_dict()

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

    model.load_state_dict(weights_rotated)

    return

def undo_mlp_permutation(model, mlp_perm, num_layers=32):
    model.to('cuda')
    weights = model.state_dict()
    for i in range(num_layers):
        weights['model.layers.'+str(i)+'.mlp.gate_proj.weight'] = weights['model.layers.'+str(i)+'.mlp.gate_proj.weight'][mlp_perm]
        weights['model.layers.'+str(i)+'.mlp.up_proj.weight'] = weights['model.layers.'+str(i)+'.mlp.up_proj.weight'][mlp_perm]
        weights['model.layers.'+str(i)+'.mlp.down_proj.weight'] = weights['model.layers.'+str(i)+'.mlp.down_proj.weight'][:,mlp_perm]

    model.load_state_dict(weights)

    return

def solve_mlp_permutation(base_model, ft_model, dataloader, n=5000, i=0, emb_size=4096):

    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)
    
    # x = torch.randn(size=(n,emb_size), dtype=torch.bfloat16).to("cuda")

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)

    # with torch.no_grad():
    #     base_model.to("cuda")
    #     y_base = base_model.model.layers[i].mlp(x)
    #     base_model.to("cpu")
        
    #     ft_model.to("cuda")
    #     y_ft = ft_model.model.layers[i].mlp(x)
    #     ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)
    
    return perm

def main():

    model_name = "meta-llama/Llama-2-7b-hf" # "EleutherAI/llemma_7b" # "meta-llama/Llama-2-7b-hf"
    rot_model_name = 'lmsys/vicuna-7b-v1.1' # "EleutherAI/llemma_7b" # "codellama/CodeLlama-7b-hf" # 'microsoft/Orca-2-7b' # "codellama/CodeLlama-7b-hf" # 'lmsys/vicuna-7b-v1.1' # "EleutherAI/llemma_7b" # 'LLM360/Amber' # 'lmsys/vicuna-7b-v1.1' # 'LLM360/Amber' # 'openlm-research/open_llama_7b' # 'lmsys/vicuna-7b-v1.1' # "lmsys/vicuna-7b-v1.1"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #bfloat16
    model_rotated = AutoModelForCausalLM.from_pretrained(rot_model_name, torch_dtype=torch.bfloat16)

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer) # prepare_random_sample_dataset(5000, 512, vocab_size)
    dataloader = prepare_hf_dataloader(dataset,1)



    rotate_model(model_rotated)
    mlp_perm = torch.arange(11008)
    permute_model(model_rotated, model_rotated, mlp_perm, torch.randperm(4096))

    mlp_perm = solve_mlp_permutation(model, model_rotated, dataloader)

    print(mlp_perm)

    undo_mlp_permutation(model, mlp_perm)

    

    # print(evaluate(model, dataloader))

    model.to('cuda')

    weights = model.state_dict()

    for i in range(32):
        weights[f'model.layers.{i}.mlp.gate_proj.weight'] = weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights[f'model.layers.{i}.mlp.up_proj.weight'] = weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(4096)

    model.load_state_dict(weights)
    model.to('cpu')

    # model.to('cpu')
    
    # print(evaluate(model, dataloader))

    model_rotated.to('cuda')

    weights_rotated = model_rotated.state_dict()

    for i in range(32):
        weights_rotated[f'model.layers.{i}.mlp.gate_proj.weight'] = weights_rotated[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights_rotated[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights_rotated[f'model.layers.{i}.mlp.up_proj.weight'] = weights_rotated[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights_rotated[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights_rotated[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(4096)

    model_rotated.load_state_dict(weights_rotated)
    model_rotated.to('cpu')

    # config = LlamaConfig()

    # model = LlamaForCausalLM(config)
    # model_rotated = LlamaForCausalLM(config)

    # print(evaluate(model, dataloader))

    

    # cshr(model, model_rotated, num_layers=num_layers, emb_size=hidden_dim)
    # act_robust(model, model_rotated, dataloader, num_layers=num_layers)
    # print("Random model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    # print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, num_layers)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with wikitext inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=32)))
    # print("mlp cshr with random canonical basis: " + str(mlp_cshr_per_layer(model, model_rotated, 0)))
    # print()

    # x = torch.randint(low=0, high=vocab_size, size=(batch_size,seq_length))

    # print("Rotated model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, 32)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with wikitext inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=32)))
    print(mlp_cshr_per_layer(model, model_rotated, 0))


if __name__ == "__main__":
    main()