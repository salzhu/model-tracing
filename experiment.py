MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import pickle
import timeit
import subprocess

from tracing.utils.llama.model import avg_model,permute_model
from tracing.utils.llama.matching import align_model
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader,evaluate

from tracing.statistics.mc import statistic as mode_stat
from tracing.statistics.cos import statistic as cos_stat

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--base_model_id',default="meta-llama/Llama-2-7b-hf",type=str)
parser.add_argument('--ft_model_id',default="lmsys/vicuna-7b-v1.1",type=str)

parser.add_argument('--permute',action='store_true')
parser.add_argument('--align',action='store_true')

parser.add_argument('--dataset_id',default="dlwh/wikitext_103_detokenized",type=str)
parser.add_argument('--block_size',default=512,type=int)
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--save',default="results.p",type=str)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--token',default="",type=str)

parser.add_argument('--stat',default="mode",type=str)
parser.add_argument('--attn',action='store_true')
parser.add_argument('--emb',action='store_true')

args = parser.parse_args()

from huggingface_hub import login
login(token=args.token)

start = timeit.default_timer()

results = {}
results['args'] =  args
results['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

torch.manual_seed(args.seed)

base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, torch_dtype=torch.bfloat16)
base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)

ft_model = AutoModelForCausalLM.from_pretrained(args.ft_model_id, torch_dtype=torch.bfloat16)
ft_tokenizer = AutoTokenizer.from_pretrained(args.ft_model_id, use_fast=False)

if args.permute is True:
    mlp_permutation = torch.randperm(MLP_SIZE)
    emb_permutation = torch.randperm(EMB_SIZE)
    permute_model(ft_model,ft_model,mlp_permutation,emb_permutation)

tmp_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, torch_dtype=torch.bfloat16)
tmp_tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)

dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized",args.block_size,base_tokenizer)
dataloader = prepare_hf_dataloader(dataset,args.batch_size)

if args.stat == "mode":
    test_stat = lambda base_model,ft_model : mode_stat(base_model,ft_model,tmp_model,dataloader,args.attn,args.emb)
if args.stat == "cos":
    test_stat = lambda base_model,ft_model : cos_stat(base_model,ft_model,N_BLOCKS)

results['base loss'] = sum(evaluate(base_model,dataloader))
results['ft loss'] = sum(evaluate(ft_model,dataloader))

results['non-aligned test stat'] = test_stat(base_model,ft_model)

if args.align is True:
    align_model(base_model,ft_model,tmp_model)
    results['aligned test stat'] = test_stat(base_model,tmp_model)

end = timeit.default_timer()
results['time'] = end - start

print(results)
pickle.dump(results,open(args.save,"wb"))
