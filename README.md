# LLM Interpolation Analysis

This repository contains code to analyze the relationship between interpolation of large language models (LLMs) and perplexity.

## Functionality

### Model Loading
Loads multiple LLMs from HuggingFace (currently supports `meta-llama/Llama-2-7b-hf`, `codellama/CodeLlama-7b-hf`, and `openlm-research/open_llama_7b`).

### L2 Distance Calculation
Calculates the L2 distance between pairs of models to quantify their differences in parameter space.

### Model Interpolation
Interpolates between pairs of models using a weighted average of their parameters, controlled by an alpha value (0.0 to 1.0).

### Perplexity Evaluation
Evaluates the perplexity of the interpolated models on the Wikitext 103 dataset.

### Visualization
Generates plots illustrating the relationship between the interpolation alpha and the resulting perplexity.

### Data Logging
Saves L2 distances and perplexity values for different alpha settings to a CSV file for further analysis.

## Requirements

A Python 3.7+ environment
Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Afterwards run the script as follows:
``` python main.py ```