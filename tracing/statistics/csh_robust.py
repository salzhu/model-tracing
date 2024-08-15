import torch
from collections import defaultdict
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel
from transformers import AutoConfig, AutoModel

import os

from tracing.utils.utils import cossim
from tracing.utils.evaluate import evaluate, evaluate_csh_test
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets

N_BLOCKS = 32

# from huggingface_hub import login
# hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]

# login(token=hf_token)

def statistic(base_model, ft_model, dataloader):
    return act_robust(base_model, ft_model, dataloader)

def statistic_rand(base_model, ft_model):
    return act_rand_mlp(base_model,ft_model)

def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def act_median_max(base_model,ft_model,dataloader,i):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_model.model.layers[i].input_layernorm.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hook)

    evaluate(base_model,dataloader)
    evaluate(ft_model,dataloader)

    base_mat = torch.vstack(feats['base']).view(-1,4096).T
    ft_mat = torch.vstack(feats['ft']).view(-1,4096).T
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def act_robust(base_model, ft_model, dataloader, num_layers=32):

    feats = defaultdict(list)

    base_hooks = {}
    ft_hooks = {}

    for i in range(num_layers):

        layer = str(i)

        base_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"base_"+layer)
        base_model.model.layers[i].input_layernorm.register_forward_hook(base_hooks[layer])
        # base_model.layers[i].input_layernorm.register_forward_hook(base_hooks[layer])

        ft_hooks[layer] = lambda m,inp,op,layer=layer,feats=feats: hook(m,inp,op,feats,"ft_"+layer)
        ft_model.model.layers[i].input_layernorm.register_forward_hook(ft_hooks[layer])
        # ft_model.layers[i].input_layernorm.register_forward_hook(ft_hooks[layer])

    evaluate_csh_test(base_model,dataloader)
    evaluate_csh_test(ft_model,dataloader)

    stats = []

    for i in range(num_layers):

        base_mat = torch.vstack(feats[f'base_{i}']).view(-1,4096).T
        ft_mat = torch.vstack(feats[f'ft_{i}']).view(-1,4096).T

        val = torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()
    
        stats.append(val)
        print(i, val)

    print(stats)
    print(sum(stats) / len(stats))

    cleanStats = [x for x in stats if str(x) != 'nan']

    return sum(cleanStats) / len(cleanStats), stats

def act_rand_mlp(base_model,ft_model,num_layers=32,n=5000,emb_size=4096,from_config=False):
    meds = []
    for i in range(num_layers):
        if not from_config:
            med = act_layer_rand(base_model, ft_model, i, n, emb_size)
        else:
            med = act_layer_rand_from_config(base_model, ft_model, i, n, emb_size)
        meds.append(med)
        print(i, med)

    print(meds)
    cleanMeds = [x for x in meds if str(x) != 'nan']
    print(sum(cleanMeds) / len(cleanMeds), meds)

    return sum(cleanMeds) / len(cleanMeds), meds


def act_layer_rand(base_model,ft_model,i,n=5000,emb_size=4096):
    feats = defaultdict(list)

    # print(base_model.state_dict().keys())
    # print(base_model.state_dict()['model.layers.8.mlp.gate_proj.weight'].shape)
    # print(base_model.state_dict()['model.layers.8.mlp.up_proj.weight'].shape)
    # print(base_model.state_dict()['model.layers.8.mlp.down_proj.weight'].shape)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    # x = torch.randn(size=(n,emb_size)).bfloat16().to("cuda")
    x = torch.randn(size=(n,emb_size), dtype=torch.bfloat16).to("cuda")
    # print(x.shape)
    with torch.no_grad():
        base_model.to("cuda")
        y_base = base_model.model.layers[i].mlp(x)
        base_model.to("cpu")
        
        ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(x)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()


def act_layer_rand_can_basis(base_model,ft_model,i,n=5000,emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    # x = torch.randn(size=(n,emb_size)).bfloat16().to("cuda")
    x = torch.randn(size=(n,1, 11008), dtype=torch.bfloat16).to("cuda")
    # print(x.shape)

    with torch.no_grad():
        base_model.to("cuda")
        y_base = base_model.model.layers[i].mlp(torch.matmul(x, base_model.model.layers[i].mlp.gate_proj.weight))
        # y_base = base_model.model.layers[i].mlp(torch.matmul(x, base_model.model.layers[i].mlp.gate_proj.weight) * base_model.model.layers[i].post_attention_layernorm.weight * base_model.model.layers[i].post_attention_layernorm.weight)
        base_model.to("cpu")
        
        ft_model.to("cuda")
        y_ft = ft_model.model.layers[i].mlp(torch.matmul(x, ft_model.model.layers[i].mlp.gate_proj.weight))
        # y_ft = ft_model.model.layers[i].mlp(torch.matmul(x, ft_model.model.layers[i].mlp.gate_proj.weight) * ft_model.model.layers[i].post_attention_layernorm.weight* ft_model.model.layers[i].post_attention_layernorm.weight)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values).item()

def act_rand_mlp_can_basis(base_model,ft_model,num_layers=32,n=5000,emb_size=4096):
    meds = []
    for i in range(num_layers):
        med = act_layer_rand_can_basis(base_model, ft_model, i, n, emb_size)
        meds.append(med)
        print(i, med)

    print(meds)
    cleanMeds = [x for x in meds if str(x) != 'nan']
    print(sum(cleanMeds) / len(cleanMeds), meds)

    return sum(cleanMeds) / len(cleanMeds), meds

def act_layer_rand_from_config(base_model,ft_model,i,n=5000,emb_size=4096):
    feats = defaultdict(list)

    # print(base_model.state_dict().keys())
    # print(base_model.state_dict()['model.layers.8.mlp.gate_proj.weight'].shape)
    # print(base_model.state_dict()['model.layers.8.mlp.up_proj.weight'].shape)
    # print(base_model.state_dict()['model.layers.8.mlp.down_proj.weight'].shape)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = base_model.layers[i].mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ft_model.layers[i].mlp.down_proj.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,emb_size)).bfloat16().to("cuda")
    # print(x.shape)
    with torch.no_grad():
        base_model.to("cuda")
        y_base = base_model.layers[i].mlp(x)
        base_model.to("cpu")
        
        ft_model.to("cuda")
        y_ft = ft_model.layers[i].mlp(x)
        ft_model.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])

    print(base_mat.shape)
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T

    print(base_mat.shape)

    # print(base_mat.shape)
    
    base_handle.remove()
    ft_handle.remove()

    print(cossim(base_mat,ft_mat).shape)
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)

if __name__ == "__main__":
    
    base_model_name = "codellama/CodeLlama-7b-hf" # 'LLM360/Amber' # "codellama/CodeLlama-7b-hf" # 'LLM360/Amber' # "codellama/CodeLlama-7b-hf"  # 'yahma/llama-7b-hf' # 'openlm-research/open_llama_7b' #  #  # "meta-llama/Llama-2-7b-hf"
    ft_model_name = "meta-llama/Llama-2-7b-hf" # 'ibm-granite/granite-7b-instruct' # "codellama/CodeLlama-7b-hf" #  # 'microsoft/Orca-2-7b' #  # "lmsys/vicuna-7b-v1.1"

    # config = AutoConfig.from_pretrained(base_model_name)
    # rnd_init_model_1 = AutoModel.from_config(config, torch_dtype=torch.bfloat16)
    # rnd_init_model_2 = AutoModel.from_config(config, torch_dtype=torch.bfloat16)
    # act_rand_mlp(rnd_init_model_1, rnd_init_model_2, from_config=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)

    base_model.to('cuda')

    base_weights = base_model.state_dict()

    for i in range(32):
        base_weights[f'model.layers.{i}.mlp.gate_proj.weight'] = base_weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(base_weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        base_weights[f'model.layers.{i}.mlp.up_proj.weight'] = base_weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(base_weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        base_weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(4096)

    base_model.load_state_dict(base_weights)
    base_model.to('cpu')

    ft_model.to('cuda')

    ft_weights = ft_model.state_dict()

    for i in range(32):
        ft_weights[f'model.layers.{i}.mlp.gate_proj.weight'] = ft_weights[f'model.layers.{i}.mlp.gate_proj.weight']@torch.diag(ft_weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        ft_weights[f'model.layers.{i}.mlp.up_proj.weight'] = ft_weights[f'model.layers.{i}.mlp.up_proj.weight']@torch.diag(ft_weights[f'model.layers.{i}.post_attention_layernorm.weight'])
        ft_weights[f'model.layers.{i}.post_attention_layernorm.weight'] = torch.ones(4096)

    ft_model.load_state_dict(ft_weights)
    ft_model.to('cpu')


    act_rand_mlp_can_basis(base_model, ft_model)

    # base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    # dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",512,base_tokenizer)
    # dataloader = prepare_hf_dataloader(dataset,1)


    # dataset = load_generated_datasets(base_model_name, ft_model_name, 512, base_tokenizer, ["text"])
    # dataloader = prepare_hf_dataloader(dataset, 1)

    # base_model = AutoModelForCausalLM.from_pretrained("/juice4/scr4/nlp/model-tracing/olmo_checkpoint/indp/seed_0_100M/")
    # ft_model = AutoModelForCausalLM.from_pretrained("/juice4/scr4/nlp/model-tracing/olmo_checkpoint/indp/seed_42_100M/")

    # print(act_layer_rand(base_model, ft_model, 10))

    # print(statistic_spcor_all(base_model, ft_model, dataloader))