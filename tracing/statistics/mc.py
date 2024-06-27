import copy

from ..utils.llama.model import avg_model
from ..utils.olmo.model import avg_model as avg_model_olmo
from ..utils.evaluate import evaluate

def statistic(base_model,ft_model,tmp_model,dataloader,attn=False,emb=False):
    if 'olmo' in base_model._get_name().lower():
        avg_model_olmo(base_model,ft_model,tmp_model,attn=attn,emb=emb)
    else:
        avg_model(base_model,ft_model,tmp_model,attn=attn,emb=emb)
    blah = evaluate(tmp_model,dataloader)
    return sum(blah), len(blah)
