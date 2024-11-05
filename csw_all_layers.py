import torch
import scipy 
from scipy.optimize import linear_sum_assignment as LAP
from transformers import AutoModelForCausalLM, AutoConfig

from tracing.utils.utils import cossim, fisher

import warnings
warnings.filterwarnings("ignore")

def csw_sp_pair(base_model,ft_model,layer_name_base, layer_name_ft):

    base_mat = base_model.state_dict()[layer_name_base]
    ft_mat = ft_model.state_dict()[layer_name_ft]

    matched = LAP(cossim(base_mat.type(torch.float64),ft_mat.type(torch.float64)), maximize=True)
    matched = matched[1]
    # csm = cossim(base_mat,ft_mat)
    # csm += (0.01)*torch.randn(csm.shape)
    # matched = torch.argmax(csm,axis=-1)
    orig = torch.arange(len(matched))

    # print(len(orig))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    return pvalue

def csw_models(base_model,ft_model):
    base_model.to('cpu')
    ft_model.to('cpu')

    weights_base = base_model.state_dict()
    weights_ft = ft_model.state_dict()

    shapes_base = {}
    shapes_ft = {}

    for name1 in list(weights_base.keys()):
        shapes_base[name1] = weights_base[name1].shape
    for name2 in list(weights_ft.keys()):
        shapes_ft[name2] = weights_ft[name2].shape

    pvalues = []

    for name1 in list(weights_base.keys()):
        for name2 in list(weights_ft.keys()):
            # print(name1,name2)
            if shapes_base[name1] == shapes_ft[name2] and len(shapes_base[name1]) != 1:
                pval = csw_sp_pair(base_model,ft_model,name1,name2)
                print(name1,name2,pval)
                pvalues.append(pval)

    # (name1, param1) in base_model.named_parameters():
    #     for (name2, param2)in ft_model.named_parameters():
    # # for (name1, param1), (name2, param2) in zip(
    # #     base_model.named_parameters(), ft_model.named_parameters()
    # # ):
    #         if param1.shape == param2.shape and param1.dim() != 1:
    #             pval = csw_sp_pair(base_model,ft_model,name1,name2)
    #             print(name1,name2,pval)
    #             pvalues.append(pval)

    print(pvalues)

    res = 0

    if len(pvalues) == 0: res = 999
    else: res = fisher(pvalues)
    
    print(res)
    return res


def main():
    model_1_id = "openai-community/gpt2" # "meta-llama/Llama-2-7b-hf"
    model_2_id = "trl-internal-testing/dummy-GPT2-correct-vocab" # "lmsys/vicuna-7b-v1.5"

    print(model_1_id, model_2_id)

    model_1 = AutoModelForCausalLM.from_pretrained(model_1_id, torch_dtype=torch.bfloat16)
    model_2 = AutoModelForCausalLM.from_pretrained(model_2_id, torch_dtype=torch.bfloat16)

    print(csw_models(model_1,model_2))

if __name__ == '__main__': main()