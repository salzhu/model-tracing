import yaml
from yaml import load, Loader

import subprocess
import argparse

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--slurm',default="nlprun -g 1 -d a6000 -r 80G -a model-tracing",type=str)
parser.add_argument('--python',default="python experiment-retrain.py",type=str)
parser.add_argument('--save',default=".",type=str)
parser.add_argument('--n_blocks',default=32,type=int)
parser.add_argument('--model_id',default="lmsys/vicuna-7b-v1.5",type=str)

args = parser.parse_args()

for layer in range(args.n_blocks):
    save_dir = f"{args.save}/layer_{layer}"
    subprocess.run(f"mkdir -p {save_dir}",shell=True)
    job_id = args.model_id.replace("/","-") + "_layer_" + layer

    log_path = f"{save_dir}/log.out"

    job = args.slurm + f" -o {log_path} -n {job_id}" \
        f" '{args.python}" + f" --model_id {args.model_id} --layer {layer} --save {save_dir}'"
    subprocess.run(job,shell=True)