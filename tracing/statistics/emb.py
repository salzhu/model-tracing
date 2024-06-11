VOCAB_SIZE = 32000

import torch

from ..utils.utils import cossim

def statistic(base_model,ft_model):
    base_mat = base_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight'][:VOCAB_SIZE]
    ft_mat = ft_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight'][:VOCAB_SIZE]
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)