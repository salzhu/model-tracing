import torch
from collections import defaultdict
import scipy
import numpy as np
from scipy.stats import chi2

from tracing.utils.utils import cossim
from tracing.utils.evaluate import evaluate

def statistic(base_model,ft_model,dataloader):
    return csh_sp_dataloader(base_model,ft_model,dataloader)

def hook(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def csh_sp_dataloader_layer(base_model,ft_model,dataloader,i):

    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,base_mat.shape[-1]).T
    ft_mat = torch.vstack(feats['ft']).view(-1,ft_mat.shape[-1]).T

    matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    return pvalue

def csh_sp_dataloader(base_model,ft_model,dataloader,num_layers=32):

    chi_squared = 0
    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].input_layernorm.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"ft_"+layer)
        ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hooks[layer])

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    p_values = []

    for i in range(num_layers):

        base_mat = torch.vstack(feats[f'base_{i}']).view(-1,base_mat.shape[-1]).T
        ft_mat = torch.vstack(feats[f'ft_{i}']).view(-1,ft_mat.shape[-1]).T

        matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
        orig = torch.arange(len(matched))

        cor, temp = scipy.stats.pearsonr(matched.tolist(), orig.tolist())

        if not np.isnan(temp):
            chi_squared -= 2 * np.log(temp)
            num_layers += 1
        p_values.append(temp)

    p_value = chi2.sf(chi_squared, df=2*num_layers)

    return p_value, p_values