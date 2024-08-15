import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ortho_group
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import scipy

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

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def statistic(base_model, ft_model, i, dataloader):
    fix_layer_norm(base_model)
    fix_layer_norm(ft_model)
    perm = mlp_matching_full_model(base_model, ft_model, dataloader)

    undo_mlp_permutation(ft_model, perm)

    return mlp_matching_per_layer(base_model, ft_model, i)

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
    
    linear_combinations_gate = torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")
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
    # print(torch.median(torch.max(mat,axis=-1).values).item())
    
    return torch.median(torch.max(mat,axis=-1).values).item()

# sends input sequences provided via dataloader through full model, uses activations from mlp i
# robust to rotation because of unrotated input
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_full_model(base_model, ft_model, dataloader, n=5000, i=0, emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    evaluate(base_model, dataloader)
    evaluate(ft_model, dataloader)
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    base_handle.remove()
    ft_handle.remove()

    perm = match_wmats(base_mat,ft_mat)
    # print(mat)
    # print(torch.median(torch.max(mat,axis=-1).values).item())
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

    model_name =  "meta-llama/Llama-2-7b-hf" # "EleutherAI/llemma_7b" 'microsoft/Orca-2-7b'
    rot_model_name = 'openlm-research/open_llama_7b' # "EleutherAI/llemma_7b" # 'openlm-research/open_llama_7b' #'LLM360/Amber'# "EleutherAI/llemma_7b" # 'lmsys/vicuna-7b-v1.1' # "EleutherAI/llemma_7b"  # 'lmsys/vicuna-7b-v1.5' # "codellama/CodeLlama-7b-hf" # 'microsoft/Orca-2-7b' 'LLM360/Amber' # 'openlm-research/open_llama_7b' 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #bfloat16
    model_rotated = AutoModelForCausalLM.from_pretrained(rot_model_name, torch_dtype=torch.bfloat16)

    rotate_model(model_rotated)
    permute_model(model_rotated, model_rotated, torch.randperm(11008), torch.randperm(4096))

    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer) # prepare_random_sample_dataset(5000, 512, vocab_size)
    dataloader = prepare_hf_dataloader(dataset,1)

    print(statistic(model, model_rotated, 0, dataloader))

    # print(model_name)
    # print(rot_model_name)

    # fix_layer_norm(model)
    # fix_layer_norm(model_rotated)

    # rotate_model(model_rotated)

    # print("Unpermuted statistic")

    # mlp_match_unperm = mlp_matching_per_layer(model, model_rotated, 0)
    # print(mlp_match_unperm)

    # orig_perm = torch.randperm(11008)
    # print("Orig perm " +  str(orig_perm))

    # rotate_model(model_rotated)
    # permute_model(model_rotated, model_rotated, orig_perm, torch.randperm(4096))

    # print("Random permuted statistic")

    # mlp_match_1 = mlp_matching_per_layer(model, model_rotated, 0)
    # print(mlp_match_1)

    # # undo mlp permutation 
    # base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer) # prepare_random_sample_dataset(5000, 512, vocab_size)
    # dataloader = prepare_hf_dataloader(dataset,1)
    # perm = mlp_matching_full_model(model, model_rotated, dataloader)
    # print(perm)

    # undo_mlp_permutation(model_rotated, perm)

    # print("Matched permutation statistic")

    # mlp_match_solved = mlp_matching_per_layer(model, model_rotated, 0)
    # print(mlp_match_solved)

if __name__ == "__main__":
    main()