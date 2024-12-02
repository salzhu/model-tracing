MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import argparse
import pickle
import timeit
import subprocess
import os

from tracing.utils.evaluate import prepare_hf_dataset, prepare_hf_dataloader, load_generated_datasets, prepare_random_sample_dataset
from tracing.utils.utils import manual_seed

from tracing.statistics.l2 import statistic as l2_stat
from tracing.statistics.jsd import statistic as jsd_stat
from tracing.statistics.csw_sp import statistic as csw_sp_stat
from tracing.statistics.csw_sp import statistic_all as csw_sp_all_stat
from tracing.statistics.csw_mm import statistic as csw_mm_stat
from tracing.statistics.csh_sp import statistic as csh_sp_stat
from tracing.statistics.csh_mm import statistic as csh_mm_stat
from tracing.statistics.csh_mm import statistic_rand as csh_mm_rand_stat
from tracing.statistics.mlp_sp import statistic as mlp_sp_stat
from tracing.statistics.mlp_sp import statistic_all as mlp_sp_all_stat
from tracing.statistics.mlp_mm import statistic as mlp_mm_stat

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--base_model_id',default="meta-llama/Llama-2-7b-hf",type=str)
parser.add_argument('--base_model_ckpt',default=3000,type=int)

parser.add_argument('--ft_model_id',default="lmsys/vicuna-7b-v1.1",type=str)
parser.add_argument('--ft_model_ckpt',default=3000,type=int)

parser.add_argument('--dataset',default="wikitext",type=str)
parser.add_argument('--block_size',default=512,type=int)
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--save',default="results.p",type=str)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--token',default="",type=str)

parser.add_argument('--stat',default="mode",type=str)

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

dtype = torch.bfloat16

base_model = GPTNeoXForCausalLM.from_pretrained(
  args.base_model_id,
  revision=f"step{args.base_model_ckpt}",
  cache_dir="./models",
)

base_tokenizer = AutoTokenizer.from_pretrained(
  args.base_model_id,
  revision=f"step{args.base_model_ckpt}",
  cache_dir="./models",
)

ft_model = GPTNeoXForCausalLM.from_pretrained(
  args.ft_model_id,
  revision=f"step{args.ft_model_ckpt}",
  cache_dir="./models",
)

ft_tokenizer = AutoTokenizer.from_pretrained(
  args.ft_model_id,
  revision=f"step{args.ft_model_ckpt}",
  cache_dir="./models",
)

print("base and ft models loaded")

if args.dataset == "wikitext":
    dataset = prepare_hf_dataset("dlwh/wikitext_103_detokenized", args.block_size, base_tokenizer)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset == "generated":
    columns_ignored = ['text']
    dataset = load_generated_datasets(args.base_model_id, args.ft_model_id, args.block_size, base_tokenizer, columns_ignored)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)
elif args.dataset == "random":
    dataset = prepare_random_sample_dataset(20, args.block_size)
    dataloader = prepare_hf_dataloader(dataset, args.batch_size)

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print("dataset loaded")

if args.stat == "csw_sp":
    test_stat = lambda base_model,ft_model : csw_sp_stat(base_model,ft_model)
if args.stat == "csw_sp_all":
    test_stat = lambda base_model,ft_model : csw_sp_all_stat(base_model,ft_model)  
if args.stat == "csh_sp":
    test_stat = lambda base_model,ft_model : csh_sp_stat(base_model,ft_model,dataloader)
    
if args.stat == "l2":
    test_stat = lambda base_model,ft_model : l2_stat(base_model,ft_model)
if args.stat == "csw_mm":
    test_stat = lambda base_model,ft_model : csw_mm_stat(base_model,ft_model,N_BLOCKS)
if args.stat == "csh_mm":
    test_stat = lambda base_model,ft_model : csh_mm_stat(base_model,ft_model,dataloader)
if args.stat == "csh_mm_rand":
    test_stat = lambda base_model,ft_model : csh_mm_rand_stat(base_model,ft_model)
if args.stat == "jsd":
    test_stat = lambda base_model,ft_model : jsd_stat(base_model,ft_model, dataloader)  

if args.stat == "mlp_sp":
    test_stat = lambda base_model,ft_model : mlp_sp_stat(base_model,ft_model,dataloader)
if args.stat == "mlp_sp_all":
    test_stat = lambda base_model,ft_model : mlp_sp_all_stat(base_model,ft_model,dataloader) 
if args.stat == "mlp_mm":
    test_stat = lambda base_model,ft_model : mlp_mm_stat(base_model,ft_model,0,dataloader)

results['test stat'] = test_stat(base_model,ft_model)

print("test stat computed")

end = timeit.default_timer()
results['time'] = end - start

print(results)
pickle.dump(results,open(args.save,"wb"))