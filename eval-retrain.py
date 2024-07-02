MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

import argparse
import pickle
import timeit
import subprocess

from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader,evaluate
from tracing.utils.llama.model import set_mlp_weights,set_weights
from tracing.statistics.mlp import statistic as stat

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--model_id',default="lmsys/vicuna-7b-v1.5",type=str)
parser.add_argument('--width_factor',default=2.0,type=float)

parser.add_argument('--ckpt',default=50000,type=int)
parser.add_argument('--load',default=".",type=str)

parser.add_argument('--dataset_id',default="dlwh/wikitext_103_detokenized",type=str)
parser.add_argument('--block_size',default=512,type=int)
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--token',default="",type=str)
parser.add_argument('--seed',default=0,type=int)

args = parser.parse_args()

from huggingface_hub import login
login(token=args.token)

start = timeit.default_timer()

torch.manual_seed(args.seed)

model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
config = AutoConfig.from_pretrained(args.model_id)

config.intermediate_size = int(args.width_factor * MLP_SIZE)
ret_model = LlamaForCausalLM(config).bfloat16()

set_weights(ret_model,model.state_dict())
for i in range(N_BLOCKS):
    mlp_state_dict = pickle.load(open(f"{args.load}/layer_{i}/ckpts/ckpt_{args.ckpt}.p","rb"))
    set_mlp_weights(ret_model,i,mlp_state_dict)

    results = pickle.load(open(f"{args.load}/layer_{i}/results.p","rb"))

    
    init_loss = results["losses"][0]
    print(f"initial loss for layer {i}: {init_loss}")
    final_loss = results["losses"][-1]
    print(f"final loss for layer {i}: {final_loss}")

print("models loaded")

dataset = prepare_hf_dataset(args.dataset_id,args.block_size,tokenizer)
dataloader = prepare_hf_dataloader(dataset,args.batch_size)

print("dataset loaded")

print(f"loss of original model: {sum(evaluate(model,dataloader))}")
print(f"loss of retrained model: {sum(evaluate(ret_model,dataloader))}")

for i in range(N_BLOCKS):
    print(f"cosine similarity of activations in layer {i}: {stat(model,ret_model,i)}")

end = timeit.default_timer()

print(f"time: {end - start}")
