import torch
from transformers import AutoModelForCausalLM

from tracing.utils.utils import cossim, spcor
import scipy 
import numpy as np
from scipy.stats import chi2

def statistic(base_model, ft_model):
    return cos_cor_fisher(base_model, ft_model)

def cos_cor_layer_2d(base_model,ft_model,layer_name):

    base_mat = base_model.state_dict()[layer_name]
    ft_mat = ft_model.state_dict()[layer_name]

    matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)

    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    return pvalue

def cos_cor_fisher(model1,model2):

    chi_squared = 0
    num_layers = 0

    p_values = []

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.dim() == 1: continue
        elif param1.shape != param2.shape:
            if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
                print(
                    f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )

        pvalue = cos_cor_layer_2d(model1, model2, name1)
        chi_squared -= 2 * np.log(pvalue)
        num_layers += 1
        p_values.append(pvalue)

    p_value = chi2.sf(chi_squared, df=2*num_layers)
    return p_value, p_values
        
if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    ft_model_name = "lmsys/vicuna-7b-v1.1" # "codellama/CodeLlama-7b-hf"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    statistic(base_model, ft_model)
