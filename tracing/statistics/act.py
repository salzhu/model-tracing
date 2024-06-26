import torch
from collections import defaultdict

from ..utils.utils import cossim
from ..utils.evaluate import evaluate

def hook(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def statistic(base_model,ft_model,dataloader,i):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[0].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[0].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,4096).T
    ft_mat = torch.vstack(feats['ft']).view(-1,4096).T
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)