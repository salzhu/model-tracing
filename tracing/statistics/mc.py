import copy

from ..utils.llama.model import avg_model
from ..utils.llama.model_70b import avg_model as avg_model_70b
from ..utils.llama.model_70b_cpu import avg_model as avg_model_70b_cpu
from ..utils.olmo.model import avg_model as avg_model_olmo
from ..utils.evaluate import evaluate, evaluate_70b

def statistic(base_model,ft_model,tmp_model,dataloader,attn=False,emb=False, alpha=0.5):
    if 'olmo' in base_model._get_name().lower():
        avg_model_olmo(base_model,ft_model,tmp_model,attn=attn,emb=emb)
    elif '70b' in base_model.name_or_path.lower():
        print(f'start average 70b model')
        avg_model = avg_model_70b_cpu(base_model,ft_model, alpha=alpha, attn=attn,emb=emb)
        print(f'end average 70b model')
        return sum(evaluate_70b(avg_model,dataloader))
    else:
        avg_model(base_model,ft_model,tmp_model,attn=attn,emb=emb)
    
    return sum(evaluate(tmp_model,dataloader))
