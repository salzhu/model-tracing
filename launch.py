import yaml
from yaml import load, Loader

import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--job_prefix',default="nlprun -g 1 -d a6000 -r 80G -a model-tracing",type=str)
parser.add_argument('--model_paths',default="config/model_list.yaml",type=str)

parser.add_argument('--save_dir',default="./",type=str)
parser.add_argument('--token',default="",type=str)
parser.add_argument('--permute',action='store_true')
parser.add_argument('--align',action='store_true')
parser.add_argument('--hi',action='store_true')

args = parser.parse_args()

model_paths = yaml.load(open(args.model_paths, 'r'), Loader=Loader)
base_models = model_paths["base_models"]
ft_models = model_paths["ft_models"]

if args.align is True:
    align = " --align"
else:
    align = ""
if args.permute is True:
    permute = " --permute"
else:
    permute = ""

if args.hi is True:
    priority = " -p high"
else:
    priority = ""

for base_model in base_models:
    for ft_model in ft_models:
        job_id = base_model.replace("/","-") + "_AND_" + ft_model.replace("/","-")

        log_path = os.path.join(args.save_dir, "logs", job_id + ".out")
        results_path = os.path.join(args.save_dir, "results", job_id + ".p")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        experiment = f"python experiment.py --ft_model_id {ft_model} --base_model_id {base_model} " \
                f"--token {args.token} --save {results_path}" \
                f"{align}{permute}"

        job = args.job_prefix + f" -o {log_path}" + f" '{experiment}'"
        print(job)
        subprocess.run(job,shell=True)
