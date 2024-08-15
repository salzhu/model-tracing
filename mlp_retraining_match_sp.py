MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP,LlamaDecoderLayer
import scipy

import numpy as np

from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader,evaluate
from tracing.utils.utils import cossim
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets, prepare_random_sample_dataset, load_dolma_programming_datasets, load_m2d2_datasets

from tracing.statistics.mlp_sp import mlp_matching_gate, mlp_matching_up

# architecture of MLP trained from scratch can be different from original
# eg, uncomment to get a 2-hidden layer MLP (original has just 1 hidden layer)
class CustomLlamaMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj1 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
#         self.gate_proj2 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj2 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj1(self.act_fn(self.gate_proj1(x)) * self.up_proj1(x))
        # down_proj = self.down_proj2(self.act_fn(self.gate_proj2(x)) * self.up_proj2(x))

        return down_proj
    
def hook(m, inp, op, feats, name):
    feats[name].append(inp[0].detach().cpu())

def main():

    model_id = "lmsys/vicuna-7b-v1.5"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(model_id)

    dataset = prepare_random_sample_dataset(20, 512)
    dataloader = prepare_hf_dataloader(dataset,1)

    i = 0 # layer to retrain
    bsz = 5000 # batch size
    T = 10000 # gradient steps
    width_fac = 2.0 # easier to get loss down for wider MLPs when retraining

    config = AutoConfig.from_pretrained(model_id)
    config.intermediate_size = int(width_fac*MLP_SIZE)

    mlp = CustomLlamaMLP(config).bfloat16()

    mlp.to("cuda")
    model.model.layers[i].mlp.to("cuda")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    A = torch.randn(size=(EMB_SIZE,EMB_SIZE),device="cuda").bfloat16() / np.sqrt(EMB_SIZE) # rotate outputs (just for kicks / sanity check)

    for t in range(T):
        X_batch = torch.randn(size=(bsz,EMB_SIZE),dtype=torch.bfloat16,device="cuda")
        with torch.no_grad():
            Y_batch = model.model.layers[i].mlp(X_batch)
            Y_batch = Y_batch@A.T
            
        Y_h = mlp(X_batch)
        
        optimizer.zero_grad()
        loss = criterion(Y_h,Y_batch)
        
        loss.backward()
        optimizer.step()
        
        if t % 100 == 0:
            print(f"train loss: {loss.item()}")

    model_retrained = LlamaForCausalLM(config).bfloat16()
    model.to('cpu')
    mlp.to('cpu')

    # Loading retrained weights to model_retrained
    weights = model.state_dict()
    for j in range(N_BLOCKS):
        gate = weights[f'model.layers.{j}.mlp.gate_proj.weight']
        gate = torch.cat((gate, torch.zeros(gate.shape)), dim=0)
        torch.reshape(gate, (config.intermediate_size, config.hidden_size))

        up = weights[f'model.layers.{j}.mlp.up_proj.weight']
        up = torch.cat((up, torch.zeros(up.shape)), dim=0)
        torch.reshape(up, (config.intermediate_size, config.hidden_size))

        down = weights[f'model.layers.{j}.mlp.down_proj.weight']
        down = torch.cat((down, torch.zeros(down.shape)), dim=-1)
        torch.reshape(down, (config.hidden_size, config.intermediate_size))

        weights[f'model.layers.{j}.mlp.gate_proj.weight'] = gate
        weights[f'model.layers.{j}.mlp.up_proj.weight'] = up
        weights[f'model.layers.{j}.mlp.down_proj.weight'] = down

    weights[f'model.layers.{i}.mlp.gate_proj.weight'] = mlp.gate_proj1.weight
    weights[f'model.layers.{i}.mlp.up_proj.weight'] = mlp.up_proj1.weight
    weights[f'model.layers.{i}.mlp.down_proj.weight'] = mlp.down_proj1.weight
    model_retrained.load_state_dict(weights)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)

    for j in range(i,N_BLOCKS):
        gate_match = mlp_matching_gate(model, model_retrained, dataloader, i=j)
        up_match = mlp_matching_up(model, model_retrained, dataloader, i=j)

        cor, pvalue = scipy.stats.pearsonr(gate_match.tolist(), up_match.tolist())
        print(j, pvalue)
        

if __name__ == "__main__":
    main()