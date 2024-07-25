import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ortho_group
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from collections import defaultdict

from tracing.utils.evaluate import evaluate, evaluate_csh_test
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets, prepare_random_sample_dataset
from tracing.statistics.cos_weights import statistic as csw
from tracing.statistics.csh_robust import act_robust
from tracing.statistics.csh_robust import act_rand_mlp as cshr
from tracing.statistics.l2 import calculate_l2_distance
from tracing.utils.utils import cossim
from tracing.statistics.cos import statistic as cswr
from tracing.statistics.jsd import statistic as jsd

torch.set_default_dtype(torch.bfloat16)

def hook_in(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def hook_out(m, inp, op, feats, name):
    feats[name].append(op.detach().cpu())

def mlp_cshr_per_layer(base_model,ft_model,i,n=5000,emb_size=4096,mlp_dim=11008):
    feats = defaultdict(list)

    base_hook = lambda *args : hook_out(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.gate_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook_out(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.gate_proj.register_forward_hook(ft_hook)

    base_model.to("cuda")
    
    linear_combinations_gate = torch.rand(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")
    linear_combinations_up = torch.rand(size=(n,1,mlp_dim), dtype=torch.bfloat16).to("cuda")

    x_base = torch.matmul(linear_combinations_gate, base_model.model.layers[i].mlp.gate_proj.weight) * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight
    x_base += torch.matmul(linear_combinations_up, base_model.model.layers[i].mlp.up_proj.weight) * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight
    x_base = torch.squeeze(x_base)

    with torch.no_grad():
        y_base = base_model.model.layers[i].mlp(x_base)
        base_model.to("cpu")

    ft_model.to("cuda")
    
    x_ft = torch.matmul(linear_combinations_gate, ft_model.model.layers[i].mlp.gate_proj.weight) * ft_model.model.layers[i].post_attention_layernorm.weight* ft_model.model.layers[i].post_attention_layernorm.weight
    x_ft += torch.matmul(linear_combinations_up, ft_model.model.layers[i].mlp.up_proj.weight) * ft_model.model.layers[i].post_attention_layernorm.weight * ft_model.model.layers[i].post_attention_layernorm.weight
    x_ft = torch.squeeze(x_ft)
    print(x_ft.shape)

    # print(x_base, x_ft)

    with torch.no_grad():    
        # ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(x_ft)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    mat = cossim(base_mat,ft_mat)

    print(mat)
    
    base_handle.remove()
    ft_handle.remove()

    # print(torch.max(cossim(base_mat,ft_mat),axis=-1).values, torch.max(cossim(base_mat,ft_mat),axis=-1).indices)
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()


def main():

    model_name = "meta-llama/Llama-2-7b-hf"
    rot_model_name = 'lmsys/vicuna-7b-v1.5' # 'LLM360/Amber' # 'openlm-research/open_llama_7b' # 'lmsys/vicuna-7b-v1.1' # "lmsys/vicuna-7b-v1.1"
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #bfloat16
    # model_rotated = AutoModelForCausalLM.from_pretrained(rot_model_name, torch_dtype=torch.bfloat16)

    config = LlamaConfig()

    model = LlamaForCausalLM(config)
    model_rotated = LlamaForCausalLM(config)

    # print(evaluate(model, dataloader))

    # base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # dataset = prepare_random_sample_dataset(5000, 512, vocab_size)
    # prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer)
    # dataloader = prepare_hf_dataloader(dataset,1)

    # cshr(model, model_rotated, num_layers=num_layers, emb_size=hidden_dim)
    # act_robust(model, model_rotated, dataloader, num_layers=num_layers)
    # print("Random model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    # print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, num_layers)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with random inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=num_layers)))
    # print("mlp cshr with random canonical basis: " + str(mlp_cshr_per_layer(model, model_rotated, 0)))
    # print()

    # x = torch.randint(low=0, high=vocab_size, size=(batch_size,seq_length))

    # print("Rotated model pair")
    # print("jsd: " + str(jsd(model, model_rotated, dataloader)))
    # print("csw spearman p-value: " + str(csw(model, model_rotated)[0]))
    # print("csw robust: " + str(cswr(model, model_rotated, num_layers)[0]))
    # print("l2: " + str(calculate_l2_distance(model, model_rotated)))
    # print("mlp cshr bad: " + str(mlp_cshr_bad(model, model_rotated, 0)))
    # print("mlp cshr with random inputs through full model: " + str(mlp_cshr_full_model(model, model_rotated, dataloader, num_layers=num_layers)))
    print(mlp_cshr_per_layer(model, model_rotated, 0))


if __name__ == "__main__":
    main()