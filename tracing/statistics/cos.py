import torch

from ..utils.utils import cossim

def statistic(base_model,ft_model,n_blocks):
    sum = 0
    for i in range(n_blocks):
        base_mat = base_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight']
        ft_mat = ft_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight']

        sum += torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)
    
    return sum