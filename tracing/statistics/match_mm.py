import torch
from collections import defaultdict

from tracing.utils.evaluate import evaluate
from tracing.utils.utils import cossim
from tracing.utils.llama.matching import match_wmats
from tracing.utils.llama.model import permute_mlp_block, fix_layer_norm

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def statistic(base_model, ft_model,dataloader,i=0):
    fix_layer_norm(base_model)
    fix_layer_norm(ft_model)
    perm = mlp_matching_full_model(base_model, ft_model, dataloader)

    permute_mlp_block(ft_model,i,ft_model,perm)

    return mlp_matching_per_layer(base_model, ft_model, i)

# sends random inputs directly (only) through mlp i
# robust to rotation because uses linear combination of mlp gates to create inputs
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_per_layer(base_model,ft_model,i,n=5000,mlp_dim=11008):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    base_model.to("cuda")
    
    linear_combinations_gate = torch.randn(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")

    x_base = torch.matmul(linear_combinations_gate, base_model.model.layers[i].mlp.gate_proj.weight)
    x_base = torch.squeeze(x_base)

    with torch.no_grad():
        y_base = base_model.model.layers[i].mlp(x_base)
        base_model.to("cpu")

    ft_model.to("cuda")
    
    x_ft = torch.matmul(linear_combinations_gate, ft_model.model.layers[i].mlp.gate_proj.weight)
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
    
    return torch.median(torch.max(mat,axis=-1).values).item()

# sends input sequences provided via dataloader through full model, uses activations from mlp i
# robust to rotation because of unrotated input
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_full_model(base_model, ft_model, dataloader, i=0):
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
