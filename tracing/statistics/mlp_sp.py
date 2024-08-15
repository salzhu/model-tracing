import torch
from collections import defaultdict
import scipy

from tracing.utils.evaluate import evaluate
from tracing.utils.llama.matching import match_wmats

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def statistic(base_model,ft_model,dataloader,n_blocks=32):

    stats = []

    for i in range(n_blocks):

        gate_match = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
        up_match = mlp_matching_up(base_model, ft_model, dataloader, i=i)

        cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
        print(i, pvalue)
        stats.append(pvalue)

    return stats

def statistic_layer(base_model,ft_model,dataloader,i=0):
    gate_perm = mlp_matching_gate(base_model, ft_model, dataloader, i=i)
    up_perm = mlp_matching_up(base_model, ft_model, dataloader, i=i)
    cor, pvalue = scipy.stats.pearsonr(gate_perm.tolist(), up_perm.tolist())
    return pvalue

# sends input sequences provided via dataloader through full model, uses activations from mlp i
# robust to rotation because of unrotated input
# prints the median of max for cosine similarity matching; and returns matched permutation 
# matching done by doing argmax of cosine similarity matrix across rows (can uncomment using LAP)
def mlp_matching_gate(base_model, ft_model, dataloader, i=0):
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

    # Alternatively: Using cosine similarity matrix and taking argmax (does not work as well)
    """
    mat = cossim(base_mat,ft_mat)
    perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    """
    
    return perm

def mlp_matching_up(base_model, ft_model, dataloader, i=0):
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

    # Alternatively: Using cosine similarity matrix and taking argmax (does not work as well)
    """
    mat = cossim(base_mat,ft_mat)
    perm = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    """
    
    return perm
