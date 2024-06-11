import yaml
from yaml import load, Loader

import subprocess
import argparse

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--slurm',default="nlprun -g 1 -d a6000 -r 80G -a model-tracing",type=str)
parser.add_argument('--python',default="python experiment.py",type=str)
parser.add_argument('--models',default="model_list.yaml",type=str)
parser.add_argument('--save',default="./",type=str)

args = parser.parse_args()

model_paths = yaml.load(open(args.models, 'r'), Loader=Loader)
base_models = model_paths["base_models"]
ft_models = model_paths["ft_models"]

for base_model in base_models:
    for ft_model in ft_models:
        job_id = base_model.replace("/","-") + "_AND_" + ft_model.replace("/","-")

        log_path = args.save + "logs/" + job_id + ".out"
        results_path = args.save + "results/" + job_id + ".p"

        job = args.slurm + f" -o {log_path} -n {job_id}" \
            f" '{args.python}" + f" --base_model_id {base_model} --ft_model_id {ft_model} --save {results_path}'"
        subprocess.run(job,shell=True)
