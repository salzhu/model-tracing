MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np

from collections import defaultdict
from tracing.utils.utils import cossim

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

def statistic(og_mlp,ret_mlp,n=5000,emb_size=4096):
    feats = defaultdict(list)

    base_hook = lambda *args : hook(*args,feats,"base")
    base_handle = og_mlp.down_proj.register_forward_hook(base_hook)

    ft_hook = lambda *args : hook(*args,feats,"ft")
    ft_handle = ret_mlp.down_proj1.register_forward_hook(ft_hook)
    
    x = torch.randn(size=(n,emb_size)).bfloat16().to("cuda")
    with torch.no_grad():
        og_mlp.to("cuda")
        y_base = og_mlp(x)
        og_mlp.to("cpu")
        
        ret_mlp.to("cuda")
        y_ft = ret_mlp(x)
        ret_mlp.to("cpu")
    
    base_mat = torch.vstack(feats['base'])
    ft_mat = torch.vstack(feats['ft'])
    
    base_mat = base_mat.view(-1,base_mat.shape[-1]).T
    ft_mat = ft_mat.view(-1,ft_mat.shape[-1]).T
    
    base_handle.remove()
    ft_handle.remove()
    
    return torch.median(torch.max(cossim(base_mat,ft_mat),axis=-1).values)

def main():
    model_id = "lmsys/vicuna-7b-v1.5"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    config = AutoConfig.from_pretrained(model_id)

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

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    print(statistic(model.model.layers[i].mlp,mlp))

if __name__ == "__main__":
    main()