import torch
from collections import defaultdict

from ..utils.utils import cossim
from ..utils.evaluate import evaluate

def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def statistic(base_model,ft_model,dataloader,i):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)