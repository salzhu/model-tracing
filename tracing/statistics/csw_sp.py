import torch

from tracing.utils.utils import cossim, fisher
from tracing.utils.llama.matching import match_wmats
import scipy 
import numpy as np
from scipy.stats import chi2

from scipy.optimize import linear_sum_assignment as LAP

def statistic(base_model, ft_model):
    return csw_sp(base_model, ft_model)

def csw_sp_layer(base_model,ft_model,layer_name):

    base_mat = base_model.state_dict()[layer_name]
    ft_mat = ft_model.state_dict()[layer_name]

    # matched = torch.argmax(cossim(base_mat,ft_mat),axis=-1)
    # matched = match_wmats(base_mat,ft_mat)
    matched = LAP(cossim(base_mat.type(torch.float64),ft_mat.type(torch.float64)), maximize=True)
    matched = matched[1]
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    return pvalue

def csw_sp(model1,model2):

    chi_squared = 0
    num_layers = 0

    p_values = []

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.dim() == 1: continue
        elif "mlp.dense_h_to_4h.weight" not in name1: continue
        elif param1.shape != param2.shape:
            print(f"{name1} shape mismatch")
            continue
            # if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
            #     print(
            #         f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
            #     )
            #     p_values.append(np.nan)
            #     continue
            # raise ValueError(
            #     f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            # )
        pvalue = csw_sp_layer(model1, model2, name1)
        if not np.isnan(pvalue):
            chi_squared -= 2 * np.log(pvalue)
            num_layers += 1
        p_values.append(pvalue)

        print(name1, pvalue)

    aggregate_pvalue = chi2.sf(chi_squared, df=2*num_layers)
    return aggregate_pvalue, p_values

def csw_sp_pair(base_model,ft_model,layer_name_base, layer_name_ft):

    base_mat = base_model.state_dict()[layer_name_base]
    ft_mat = ft_model.state_dict()[layer_name_ft]

    matched = LAP(cossim(base_mat.type(torch.float64),ft_mat.type(torch.float64)), maximize=True)
    matched = matched[1]
    orig = torch.arange(len(matched))

    cor, pvalue = scipy.stats.pearsonr(matched.tolist(), orig.tolist())
    return pvalue

def statistic_all(base_model,ft_model):
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
            if shapes_base[name1] == shapes_ft[name2] and len(shapes_base[name1]) != 1:
                pval = csw_sp_pair(base_model,ft_model,name1,name2)
                print(name1,name2,pval)
                pvalues.append(pval)

    print(pvalues)

    res = 0

    if len(pvalues) == 0: res = 999
    else: res = fisher(pvalues)
    
    print(res)
    return res
