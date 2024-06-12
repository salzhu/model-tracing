import copy

from ..utils.llama.model import avg_model
from ..utils.evaluate import evaluate

def statistic(base_model,ft_model,tmp_model,dataloader,attn=False,emb=False):
    avg_model(base_model,ft_model,tmp_model,attn=attn,emb=emb)
    return sum(evaluate(tmp_model,dataloader))
