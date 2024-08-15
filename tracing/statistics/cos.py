import torch
from transformers import AutoModelForCausalLM

from tracing.utils.utils import cossim

def statistic(base_model,ft_model,n_blocks):
    sum = 0
    all = []
    for i in range(n_blocks):
        base_mat = base_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight'].T
        ft_mat = ft_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight'].T

        sum += torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()
        all.append(torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item())
    
    return sum / len(all), all

def cos_matrices(matrix_1, matrix_2):
    # print(matrix_1.shape)
    # print(matrix_2.shape)
    # print(torch.max(cossim(matrix_1,matrix_2),axis=-1).values)
    # print(cossim(matrix_1,matrix_2))
    # print(matrix_1)
    # print(matrix_2)

    for i in range(len(matrix_1)):
        if torch.linalg.norm(matrix_1[i]) == 0: 
            print(i)

    print("---")
    
    for i in range(len(matrix_2)):
        if torch.linalg.norm(matrix_2[i]) == 0: 
            print(i)

    return torch.median(torch.max(cossim(matrix_1,matrix_2),axis=-1).values).item()

if __name__ == "__main__":

    # mat1 = torch.rand(11008, 4096)
    # mat2 = torch.rand(11008, 4096)

    # print(cos_matrices(mat1, mat2))

    base_model_name = "meta-llama/Llama-2-7b-hf" # 'openlm-research/open_llama_7b' # 'lmsys/vicuna-7b-v1.5' 
    ft_model_name = "lmsys/vicuna-7b-v1.1" # "codellama/CodeLlama-7b-hf" # 'openlm-research/open_llama_7b_v2' # 'LLM360/Amber' # 

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    # print(statistic(base_model, ft_model, 32))

    # base_model = AutoModelForCausalLM.from_pretrained("/juice4/scr4/nlp/model-tracing/olmo_checkpoint/indp/seed_0_100M/")
    # ft_model = AutoModelForCausalLM.from_pretrained("/juice4/scr4/nlp/model-tracing/olmo_checkpoint/indp/seed_42_100M/")
    print(statistic(base_model, ft_model, 32))

    # print(cos_matrices(base_model.state_dict()['model.layers.0.mlp.gate_proj.weight'].T, ft_model.state_dict()['model.layers.0.mlp.gate_proj.weight'].T))
    # print(cos_matrices(base_model.state_dict()['model.embed_tokens.weight'].T, ft_model.state_dict()['model.embed_tokens.weight'].T))
    # print(cos_matrices(base_model.state_dict()['lm_head.weight'].T, ft_model.state_dict()['lm_head.weight'].T))