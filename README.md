
# LLM Model Tracing
This repository investigates model tracing in large language models (LLMs). 

Specifically, given a base LLM and a fine-tuned LLM, this code provides functionality to:

- Permute the weights of one model (either MLP or embedding weights).
- Align the weights of the fine-tuned model to the base model using the Hungarian algorithm.
- Evaluate the effect of weight permutation and alignment on different statistics:
    - Mode connectivity
    - Cosine similarity
    - Embedding similarity
- Evaluate the perplexity of the base and fine-tuned models on a given dataset.

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

For development, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Usage

The repository provides three main scripts:

- `main.py`: Executes the main experiment pipeline for model tracing.
- `generate.py`: Generates text using specified language models and saves the output to JSON files.
- `launch.py`: Launches multiple experiments in parallel using slurm.

### `main.py`

This script performs the following steps:

1. Loads the base and fine-tuned LLMs.
2. Optionally permutes the weights of the fine-tuned model.
3. Calculates the selected statistic for the non-aligned models.
4. Optionally aligns the weights of the fine-tuned model to the base model.
5. Calculates the selected statistic for the aligned models.
6. Optionally evaluates the perplexity of the base and fine-tuned models.
7. Saves the results to a pickle file.

The script accepts various command-line arguments:

- `--base_model_id`: HuggingFace model ID for the base model.
- `--ft_model_id`: HuggingFace model ID for the fine-tuned model.
- `--permute`: Whether to permute the weights of the fine-tuned model.
- `--align`: Whether to align the weights of the fine-tuned model to the base model.
- `--dataset_id`: HuggingFace dataset ID for perplexity evaluation.
- `--stat`: Statistic to calculate (options: "mode", "cos", "emb").
- `--attn`: Whether to consider attention weights in the "mode" statistic.
- `--emb`: Whether to consider embedding weights in the "mode" statistic.
- `--eval`: Whether to evaluate perplexity.
- `--save`: Path to save the results pickle file.

Example usage:

```bash
python main.py --base_model_id meta-llama/Llama-2-7b-hf --ft_model_id lmsys/vicuna-7b-v1.1 --permute --align --dataset wikitext --stat mode --attn --save results.p
```

### `generate.py`

This script generates text using specified language models and saves the generated text to a JSON file.

### `launch.py`

This script launches multiple experiments in parallel using slurm. It reads model IDs from a YAML file and runs `main.py` for each pair of base and fine-tuned models.

## Configuration

The `model-tracing/config/model_list.yaml` file defines the base and fine-tuned models for the experiments. 
## Data

The code downloads and uses the Wikitext 103 dataset for perplexity evaluation.

## Results

The results of the experiments are saved as pickle files. The files contain dictionaries with the following keys:

- `args`: Command-line arguments used for the experiment.
- `commit`: Git commit hash of the code used for the experiment.
- `non-aligned test stat`: Value of the selected statistic for the non-aligned models.
- `aligned test stat`: Value of the selected statistic for the aligned models (if `--align` is True).
- `base loss`: Perplexity of the base model on the evaluation dataset (if `--eval` is True).
- `ft loss`: Perplexity of the fine-tuned model on the evaluation dataset (if `--eval` is True).
- `time`: Total execution time of the experiment.

## Sample commands

### 70B runs
```
 python main.py --base_model_id meta-llama/Llama-2-70b-hf --ft_model_id meta-llama/Meta-Llama-3-70B
```

### accelerate:
```
 accelerate launch --main_process_port 0  main.py --base_model_id meta-llama/Llama-2-70b-chat-hf --ft_model_id meta-llama/Llama-2-70b-chat-hf
```