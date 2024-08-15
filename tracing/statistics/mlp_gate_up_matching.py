import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ortho_group
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import scipy
import numpy as np
from scipy.stats import chi2

from tracing.utils.evaluate import evaluate, evaluate_csh_test
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets, prepare_random_sample_dataset, load_dolma_programming_datasets, load_m2d2_datasets
from tracing.statistics.cos_weights import statistic as csw
from tracing.utils.utils import cossim
from tracing.utils.llama.model import permute_model
from tracing.utils.llama.matching import match_wmats

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def statistic(base_model, ft_model, dataloader, num_layers=32, emb_dim=4096):

    stats = []

    for i in range(num_layers):

        gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
        # print(gate_match)

        up_match = mlp_matching_up(base_model, ft_model, dataloader, i=i)
        # print(up_match)

        cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
        print(i, pvalue)
        stats.append(pvalue)

    # gate_perms = mlp_gate_perms(base_model, ft_model, dataloader, num_layers, emb_dim)
    # up_perms = mlp_up_perms(base_model, ft_model, dataloader, num_layers, emb_dim)

    # stats = []

    # for i in range(num_layers):
    #     cor, pvalue = scipy.stats.pearsonr(gate_perms[i].tolist(), up_perms[i].tolist())
    #     print(i, pvalue)
    #     stats.append(pvalue)

    print(stats)
    print(fisher(stats))

    return fisher(stats)

def mlp_gate_perms(base_model, ft_model, dataloader, num_layers=32, emb_dim=4096):
    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    # print("1")

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"base_gate_"+layer)
        base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"ft_gate_"+layer)
        ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hooks[layer])

    # print("2")

    evaluate(base_model,dataloader)
    # print("3")
    evaluate(ft_model,dataloader)

    # print("4")

    gate_perms = []
    for i in range(num_layers):
        base_mat = torch.vstack(feats[f'base_gate_{i}']).view(-1,emb_dim).T
        ft_mat = torch.vstack(feats[f'ft_gate_{i}']).view(-1,emb_dim).T
        gate_perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
        # gate_perm = match_wmats(base_mat,ft_mat)

        gate_perms.append(gate_perm)

    del feats

    return gate_perms

def mlp_up_perms(base_model, ft_model, dataloader, num_layers=32, emb_dim=4096):
    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    # print("1")

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"base_up_"+layer)
        base_model.model.layers[i].mlp.up_proj.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook_out(m,inp,op,feats,"ft_up_"+layer)
        ft_model.model.layers[i].mlp.up_proj.register_forward_hook(ft_hooks[layer])

    # print("2")

    evaluate(base_model,dataloader)
    # print("3")
    evaluate(ft_model,dataloader)

    # print("4")

    up_perms = []
    for i in range(num_layers):
        base_mat = torch.vstack(feats[f'base_up_{i}']).view(-1,emb_dim).T
        ft_mat = torch.vstack(feats[f'ft_up_{i}']).view(-1,emb_dim).T
        up_perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)

        # up_perm = match_wmats(base_mat,ft_mat)

        up_perms.append(up_perm)

    del feats

    return up_perms

def fisher(pvalues):
    chi_squared = 0
    num_layers = 0
    for pvalue in pvalues:
        if not np.isnan(pvalue):
            chi_squared -= 2 * np.log(pvalue)
            num_layers += 1

    return chi2.sf(chi_squared, df=2*num_layers)

def statistic_layer(base_model, ft_model, dataloader, i=0):
    gate_perm = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
    up_perm = mlp_matching_up(base_model, ft_model, dataloader, i=i)
    cor, pvalue = scipy.stats.pearsonr(gate_perm.tolist(), up_perm.tolist())
    print(pvalue)
    return pvalue

# sends random inputs directly (only) through mlp i
# robust to rotation because uses linear combination of mlp gates to create inputs
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_per_layer(base_model,ft_model,i,n=5000,emb_size=4096,mlp_dim=11008):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    base_model.to("cuda")
    
    linear_combinations_gate = 100 * torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")
    linear_combinations_up = torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")

    x_base = torch.matmul(linear_combinations_gate, base_model.model.layers[i].mlp.gate_proj.weight) # * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight
    x_base = torch.squeeze(x_base)

    with torch.no_grad():
        y_base = base_model.model.layers[i].mlp(x_base)
        base_model.to("cpu")

    ft_model.to("cuda")
    
    x_ft = torch.matmul(linear_combinations_gate, ft_model.model.layers[i].mlp.gate_proj.weight) # * ft_model.model.layers[i].post_attention_layernorm.weight* ft_model.model.layers[i].post_attention_layernorm.weight
    x_ft = torch.squeeze(x_ft)

    with torch.no_grad():    
        y_ft = ft_model.model.layers[i].mlp(x_ft)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    mat = cossim(base_mat,ft_mat).to('cpu')
    
    base_handle.remove()
    ft_handle.remove()
    perm = torch.argmax(mat,axis=-1)
    print(torch.median(torch.max(mat,axis=-1).values).item())
    
    return perm

# sends input sequences provided via dataloader through full model, uses activations from mlp i
# robust to rotation because of unrotated input
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_gate(base_model, ft_model, dataloader, i=0, emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    base_mat.to('cuda')
    ft_mat.to('cuda')
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)
    # mat = cossim(base_mat,ft_mat)
    # print(mat)
    # print(torch.median(torch.max(mat,axis=-1).values).item())
    # print(cossim(base_mat,ft_mat))
    # perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    
    return perm

def mlp_matching_up(base_model, ft_model, dataloader, i=0, emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.up_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.up_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    base_mat.to('cuda')
    ft_mat.to('cuda')
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)
    # mat = cossim(base_mat,ft_mat)
    # print(mat)
    # print(torch.median(torch.max(mat,axis=-1).values).item())
    # print(cossim(base_mat,ft_mat))
    # perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    
    return perm

# makes all post attention layer norms have value = 1 by multiplying mlp gate_proj and up_projs
def fix_layer_norm(model, num_layers=32, hidden_dim=4096):
    model.to('cuda')

    weights = model.state_dict()

    for i in range(num_layers):
        weights[f'model.layers.{i}.mlp.gate_proj.weight'] = weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights[f'model.layers.{i}.mlp.up_proj.weight'] = weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(hidden_dim)

    model.load_state_dict(weights)
    model.to('cpu')

# randomly rotates model
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

def scale_model_mlp(model, num_layers=32):
    model.to('cuda')
    weights = model.state_dict()
    for i in range(num_layers):
        a = torch.randn(1).to('cuda')
        weights[f'model.layers.{i}.mlp.up_proj.weight'] = a * weights[f'model.layers.{i}.mlp.up_proj.weight']
        weights[f'model.layers.{i}.mlp.down_proj.weight'] = 1/a * weights[f'model.layers.{i}.mlp.down_proj.weight']

    model.load_state_dict(weights)
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

def main():

    model_name = "LLM360/Amber" # "meta-llama/Llama-2-7b-hf" # #  # 'LLM360/Amber' # "EleutherAI/llemma_7b" # 'lmsys/vicuna-7b-v1.1' 'microsoft/Orca-2-7b'
    rot_model_name = "ibm-granite/granite-7b-base" # "EleutherAI/llemma_7b" #  # "meta-llama/Llama-2-7b-hf" # "codellama/CodeLlama-7b-hf" #  # 'openlm-research/open_llama_7b' # 'ibm-granite/granite-7b-base' # # 'microsoft/Orca-2-7b' # 'LLM360/Amber' # 'lmsys/vicuna-7b-v1.1' # 'microsoft/Orca-2-7b' # "EleutherAI/llemma_7b" # 'lmsys/vicuna-7b-v1.1' # "EleutherAI/llemma_7b" # 'openlm-research/open_llama_7b' #'LLM360/Amber
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #bfloat16
    model_rotated = AutoModelForCausalLM.from_pretrained(rot_model_name, torch_dtype=torch.bfloat16)

    print(model_name)
    print(rot_model_name)
    # print("Random configs ...")

    # config = LlamaConfig()

    # model = LlamaForCausalLM(config)
    # model_rotated = LlamaForCausalLM(config)

    # fix_layer_norm(model)
    # fix_layer_norm(model_rotated)

    rotate_model(model_rotated)
    permute_model(model_rotated, model_rotated, torch.randperm(11008), torch.randperm(4096))

    # print("Model rotated and permuted...")

    # scale_model_mlp(model_rotated)

    # print("Model mlps randomly scaled...")

    print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer) # prepare_random_sample_dataset(5000, 512, vocab_size)
    dataset = prepare_random_sample_dataset(20, 512)
    # columns_ignored = ['text', 'added', 'id', 'lang', 'metadata', 'source', 'timestamp', 'subdomain']
    # dataset = load_dolma_programming_datasets("cpp", 512, base_tokenizer, columns_ignored)
    # columns_ignored = ['text', 'added', 'id', 'source', 'subdomain']
    # dataset = load_m2d2_datasets("IM", 512, base_tokenizer, columns_ignored)
    dataloader = prepare_hf_dataloader(dataset,1)

    # print(evaluate(model, dataloader))
    # print(evaluate(model_rotated, dataloader))

    # print(statistic(model, model_rotated, dataloader))

    for i in range(32):

        gate_match = mlp_matching_gate(model, model_rotated, dataloader, i=i)
        # print(gate_match)

        up_match = mlp_matching_up(model, model_rotated, dataloader, i=i)
        # print(up_match)

        cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
        print(i, pvalue)

if __name__ == "__main__":
    main()