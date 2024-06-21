import torch
from transformers import AutoModelForCausalLM

from tracing.utils.utils import cossim, spcor
import scipy 

def cos_cor_pvalue(base_model,ft_model,layer_name):

    base_mat = base_model.state_dict()[layer_name]
    ft_mat = ft_model.state_dict()[layer_name]

    matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)

    # print(matched)

    orig = torch.arange(len(matched))
    # print(orig)

    cor = spcor(matched, orig)
    # print(cor)

    cor2, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    # print(cor2, pvalue)

    return cor2, pvalue


if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    ft_model_name = "codellama/CodeLlama-7b-hf"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    cos_cor_pvalue(base_model, ft_model, 'model.layers.'+str(31)+'.mlp.gate_proj.weight')
