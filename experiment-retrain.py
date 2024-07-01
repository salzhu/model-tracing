MLP_SIZE = 11008
EMB_SIZE = 4096
N_BLOCKS = 32

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaMLP

import argparse
import pickle
import timeit
import subprocess

from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader,evaluate

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--model_id',default="lmsys/vicuna-7b-v1.5",type=str)
parser.add_argument('--layer',default=0,type=int)
parser.add_argument('--width_factor',default=2.0,type=float)
parser.add_argument('--batch_size',default=1024,type=int)
parser.add_argument('--n_batches',default=100000,type=int)
parser.add_argument('--learning_rate',default=0.001,type=float)
parser.add_argument('--save',default=".",type=str)
parser.add_argument('--token',default="",type=str)

args = parser.parse_args()

from huggingface_hub import login
login(token=args.token)

start = timeit.default_timer()

results = {}
results['args'] =  args
results['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

torch.manual_seed(args.seed)

model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
config = AutoConfig.from_pretrained(args.model_id)
config.intermediate_size = int(args.width_factor * MLP_SIZE)

print("model loaded")

teacher = model.model.layers[args.layer].mlp.to("cuda")
student = LlamaMLP(config).to("cuda")

print("starting training")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)

results["losses"] = []
subprocess.run(f"mkdir -p {args.save}/ckpts",shell=True)

for t in range(1,args.n_batches+1):
    X_batch = torch.randn(size=(args.batch_size,MLP_SIZE),dtype=torch.bfloat16,device="cuda")
    with torch.no_grad():
        Y_batch = teacher(X_batch)
    
    Y_h = student(X_batch)
    loss = criterion(Y_h,Y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 500 == 0:
        print(f"train loss: {loss.item()}")
        results["losses"].append(loss.item())
    if t % 10000 == 0:
        pickle.dump(student.state_dict(),open(f"{args.save}/ckpts/ckpt_{t}.p"))

end = timeit.default_timer()
results['time'] = end - start

print(results)
pickle.dump(results,open(args.save,"wb"))
