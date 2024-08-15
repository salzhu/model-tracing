import torch
from collections import defaultdict

from tracing.utils.utils import cossim
from tracing.utils.evaluate import evaluate

def statistic(base_model,ft_model,dataloader):
    return csh_mm_dataloader(base_model, ft_model, dataloader)

def statistic_rand(base_model,ft_model):
    return csh_mm_rand_cbasis(base_model,ft_model)

def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def csh_mm_dataloader_block(base_model,ft_model,dataloader,i):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,base_mat.shape[-1]).T
    ft_mat = torch.vstack(feats['ft']).view(-1,ft_mat.shape[-1]).T
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def csh_mm_dataloader(base_model,ft_model,dataloader,n_blocks=32):

    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(n_blocks):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].input_layernorm.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"ft_"+layer)
        ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hooks[layer])

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    stats = []

    for i in range(n_blocks):

        base_mat = torch.vstack(feats[f'base_{i}']).view(-1,base_mat.shape[-1]).T
        ft_mat = torch.vstack(feats[f'ft_{i}']).view(-1,ft_mat.shape[-1]).T

        val = torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()
    
        stats.append(val)
        print(i, val)

    cleanStats = [x for x in stats if str(x) != 'nan']

    return sum(cleanStats) / len(cleanStats), stats

def csh_mm_rand(base_model,ft_model,n_blocks=32,n=5000,emb_size=4096):
    meds = []
    for i in range(n_blocks):
        med = csh_mm_rand_block(base_model, ft_model, i, n, emb_size)
        meds.append(med)
        print(i, med)

    cleanMeds = [x for x in meds if str(x) != 'nan']

    return sum(cleanMeds) / len(cleanMeds), meds

def csh_mm_rand_block(base_model,ft_model,i,n=5000,emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,emb_size), dtype=torch.bfloat16).to("cuda")
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
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def csh_mm_rand_cbasis_block(base_model,ft_model,i,n=5000,mlp_size=11008):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,1,mlp_size), dtype=torch.bfloat16).to("cuda")

    with torch.no_grad():
        base_model.to("cuda")
        y_base = base_model.model.layers[i].mlp(torch.matmul(x, base_model.model.layers[i].mlp.gate_proj.weight))
        base_model.to("cpu")
        
        ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(torch.matmul(x, ft_model.model.layers[i].mlp.gate_proj.weight))
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def csh_mm_rand_cbasis(base_model,ft_model,n_blocks=32,n=5000,emb_size=4096):
    meds = []
    for i in range(n_blocks):
        med = csh_mm_rand_cbasis_block(base_model, ft_model, i, n, emb_size)
        meds.append(med)
        print(i, med)

    cleanMeds = [x for x in meds if str(x) != 'nan']

    return sum(cleanMeds) / len(cleanMeds), meds
