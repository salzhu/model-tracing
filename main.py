MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast

import argparse
import pickle
import timeit
import subprocess
import os

from tracing.utils.llama.model import permute_model
from tracing.utils.olmo.model import permute_model as permute_model_olmo
from tracing.utils.llama.matching import align_model
from tracing.utils.evaluate import prepare_hf_dataset, prepare_aya_dataset, prepare_hf_dataloader,evaluate, load_dolma_programming_datasets, load_m2d2_datasets, load_generated_datasets, prepare_random_sample_dataset
from tracing.utils.utils import manual_seed

from tracing.statistics.mc import statistic as mode_stat
from tracing.statistics.cos import statistic as cos_stat
from tracing.statistics.emb import statistic as emb_stat
from tracing.statistics.l2 import statistic as l2_stat
from tracing.statistics.jsd import statistic as jsd_stat
from tracing.statistics.cos_weights import statistic as cos_weight_stat
from tracing.statistics.act import statistic as cos_act_stat
from tracing.statistics.csh_robust import statistic as csh_robust_stat
from tracing.statistics.csh_robust import statistic_rand as csh_robust_rand_stat
from scripts.perm.main import statistic as perm_mc_l2_stat

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--base_model_id',default="meta-llama/Llama-2-7b-hf",type=str)
parser.add_argument('--ft_model_id',default="lmsys/vicuna-7b-v1.1",type=str)

parser.add_argument('--permute',action='store_true')
parser.add_argument('--align',action='store_true')

parser.add_argument('--dataset',default="wikitext",type=str)
parser.add_argument('--block_size',default=512,type=int)
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--save',default="results.p",type=str)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--token',default="",type=str)

parser.add_argument('--stat',default="mode",type=str)
parser.add_argument('--attn',action='store_true')
parser.add_argument('--emb',action='store_true')
parser.add_argument('--num_perm',default=99,type=int)

parser.add_argument('--eval',action='store_true')

parser.add_argument('--aya_subset', default="aya_human_annotated", type=str, help="Subset of Aya dataset")
parser.add_argument('--aya_language', default="eng", type=str, help="Language code for Aya dataset")

args = parser.parse_args()

from huggingface_hub import login
if args.token != "":
    hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
else:
    hf_token = args.token
login(token=hf_token)

start = timeit.default_timer()

results = {}
results['args'] =  args
results['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# fix seed on torch, np and random
manual_seed(args.seed)

base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, torch_dtype=torch.bfloat16)
if 'olmo' in args.base_model_id.lower():
    tokenizer_name = 'allenai/OLMo-1.7-7B-hf' if 'olmo' in args.base_model_id.lower() else args.base_model_id
    base_tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name, use_fast=False)
elif 'Alfred' in args.base_model_id:
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
elif 'Salesforce' in args.base_model_id:
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id,trust_remote_code=True)
else:
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)

ft_model = AutoModelForCausalLM.from_pretrained(args.ft_model_id, torch_dtype=torch.bfloat16)
if 'olmo' in args.ft_model_id.lower():
    tokenizer_name = 'allenai/OLMo-1.7-7B-hf' if 'olmo' in args.ft_model_id.lower() else args.ft_model_id
    ft_tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name, use_fast=False)
elif 'Alfred' in args.ft_model_id:
    ft_tokenizer = AutoTokenizer.from_pretrained(args.ft_model_id)
elif 'Salesforce' in args.ft_model_id:
    ft_tokenizer = AutoTokenizer.from_pretrained(args.ft_model_id,trust_remote_code=True)
else:
    ft_tokenizer = AutoTokenizer.from_pretrained(args.ft_model_id, use_fast=False)

print("base and ft models loaded")

if args.permute is True:
    mlp_permutation = torch.randperm(MLP_SIZE)
    emb_permutation = torch.randperm(EMB_SIZE)
    if 'olmo' in args.base_model_id.lower():
        permute_model_olmo(base_model,ft_model,mlp_permutation,emb_permutation)
    else:
        permute_model(ft_model,ft_model,mlp_permutation,emb_permutation)
    print("ft model permuted")

tmp_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, torch_dtype=torch.bfloat16)
if 'olmo' in args.base_model_id.lower():
    tmp_tokenizer = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name, use_fast=False)
elif 'Alfred' in args.base_model_id:
    tmp_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
elif 'Salesforce' in args.base_model_id:
    tmp_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id,trust_remote_code=True)
else:
    tmp_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)

if 'Salesforce' in args.base_model_id: 
    base_tokenizer = ft_tokenizer

if args.dataset == "wikitext":
    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized", args.block_size, base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset == "aya":
    dataset = prepare_aya_dataset(args.aya_subset, args.aya_language, args.block_size, base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset.startswith("dolma_"):
    language = args.dataset.split("_")[1]
    if not language and language is not None:
        raise ValueError("Language is an empty string")
    columns_ignored = ['text', 'added', 'id', 'lang', 'metadata', 'source', 'timestamp', 'subdomain']
    dataset = load_dolma_programming_datasets(language, args.block_size, base_tokenizer, columns_ignored)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset.startswith("m2d2_"):
    test_case = args.dataset.split("_")[1]
    if not test_case:
        raise ValueError("Invalid m2d2 dataset format. Use 'm2d2_testcase' (e.g., 'm2d2_AI')")
    columns_ignored = ['text', 'added', 'id', 'source', 'subdomain']
    dataset = load_m2d2_datasets(test_case, args.block_size, base_tokenizer, columns_ignored)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset == "generated":
    columns_ignored = ['text']
    dataset = load_generated_datasets(args.base_model_id, args.ft_model_id, args.block_size, base_tokenizer, columns_ignored)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset == "random":
    dataset = prepare_random_sample_dataset(1, args.block_size)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print("dataset loaded")

if args.stat == "mode":
    test_stat = lambda base_model,ft_model : mode_stat(base_model,ft_model,tmp_model,dataloader,args.attn,args.emb)
if args.stat == "csw_spearman":
    test_stat = lambda base_model,ft_model : cos_weight_stat(base_model,ft_model)
if args.stat == "csh_spearman":
    test_stat = lambda base_model,ft_model : cos_act_stat(base_model,ft_model,dataloader)

if args.stat == "l2":
    test_stat = lambda base_model,ft_model : l2_stat(base_model,ft_model)
if args.stat == "csw_robust":
    test_stat = lambda base_model,ft_model : cos_stat(base_model,ft_model,N_BLOCKS)
if args.stat == "csh_robust":
    test_stat = lambda base_model,ft_model : csh_robust_stat(base_model,ft_model,dataloader)
if args.stat == "csh_robust_mlp_rand":
    test_stat = lambda base_model,ft_model : csh_robust_rand_stat(base_model,ft_model)
if args.stat == "jsd":
    test_stat = lambda base_model,ft_model : jsd_stat(base_model,ft_model, dataloader)
if args.stat == "emb":
    test_stat = lambda base_model,ft_model : emb_stat(base_model,ft_model)

if args.stat == "perm_mc_l2":
    mc = lambda base_model,ft_model : mode_stat(base_model,ft_model,tmp_model,dataloader,args.attn,args.emb)
    l2 = lambda base_model,ft_model : l2_stat(base_model,ft_model)
    test_stat = lambda base_model,ft_model : perm_mc_l2_stat(base_model,ft_model,mc,l2,args.num_perm)

if args.eval is True:
    results['base loss'] = sum(evaluate(base_model,dataloader))
    results['ft loss'] = sum(evaluate(ft_model,dataloader))
    print("losses evaluated")

results['non-aligned test stat'] = test_stat(base_model,ft_model)

print("non-aligned stat computed")

if args.align is True:
    align_model(base_model,ft_model,ft_model)
    results['aligned test stat'] = test_stat(base_model,ft_model)
    print("aligned stat computed")

end = timeit.default_timer()
results['time'] = end - start

print(results)
pickle.dump(results,open(args.save,"wb"))