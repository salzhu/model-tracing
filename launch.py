import yaml
from yaml import load, Loader

import subprocess
import argparse

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--slurm',default="nlprun -g 1 -d a6000 -r 80G -a model-tracing",type=str)
parser.add_argument('--python',default="python experiment.py",type=str)
parser.add_argument('--model_paths',default="model_list.yaml",type=str)
parser.add_argument('--save_dir',default="./",type=str)

args = parser.parse_args()

model_paths = yaml.load(open(args.model_paths, 'r'), Loader=Loader)
base_models = model_paths["base_models"]
ft_models = model_paths["ft_models"]

for base_model in base_models:
    for ft_model in ft_models:
        job_id = base_model.replace("/","-") + "_AND_" + ft_model.replace("/","-")

        log_path = args.save_dir + "logs/" + job_id + ".out"
        results_path = args.save_dir + "results/" + job_id + ".p"

        job = args.slurm + f" -o {log_path} " + args.python + f" --save {results_path}"
        subprocess.run(job,shell=True)
