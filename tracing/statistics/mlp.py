import torch
from collections import defaultdict

from ..utils.utils import cossim
from ..utils.evaluate import evaluate

def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def statistic(base_model,ft_model,i,n=5000,emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,emb_size)).bfloat16().to("cuda")
    with torch.no_grad():
        base_model.to("cuda")
        y_base = base_model.model.layers[i].mlp(x)
        base_model.to("cpu")
        
        ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(x)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)