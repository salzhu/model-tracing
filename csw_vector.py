import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import scipy
import yaml
from yaml import load, Loader

import faiss
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class WeightInfo:
    model_name: str
    param_name: str
    shape: Tuple[int, ...]
    
class CSWSearch:
    def __init__(self):
        # Separate index for each dimension combination
        self.indices: Dict[Tuple[int, int], faiss.Index] = {}
        # Keep track of what each index position corresponds to
        self.metadata: Dict[Tuple[int, int], List[WeightInfo]] = {}
        
    def add_weight_matrix(self, model_name: str, 
                         param_name: str, weight_matrix: np.ndarray) -> None:
        """Add a weight matrix while preserving dimensional information"""
        d1, d2 = weight_matrix.shape
        dim_key = (d1, d2)
        
        # First time seeing this dimension combination
        if dim_key not in self.indices:
            # Create new index for this dimension
            self.indices[dim_key] = faiss.IndexFlatIP(d1 * d2)
            self.metadata[dim_key] = []
            
        # Flatten matrix in row-major order and normalize
        flat_weights = np.array(weight_matrix.reshape(1, -1)).astype(np.float32)
        faiss.normalize_L2(flat_weights)  # for cosine similarity
        
        # Add to appropriate index
        self.indices[dim_key].add(flat_weights)
        
        # Store metadata
        self.metadata[dim_key].append(
            WeightInfo(model_name, param_name, (d1, d2))
        )
    
    def find_similar_weights(self, model_name: str, weight_matrix: np.ndarray, 
                           k: int = 5) -> List[Tuple[WeightInfo, float]]:
        """Find similar weight matrices, but only among those with matching dimensions"""
        d1, d2 = weight_matrix.shape
        dim_key = (d1, d2)
        
        if dim_key not in self.indices:
            raise ValueError(f"No weight matrices found with dimensions {dim_key}")
        
        # Prepare query in same way as stored matrices
        query = np.array(weight_matrix.reshape(1, -1)).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        D, I = self.indices[dim_key].search(query, k + 1)  # +1 for self-match
        
        # Format results (excluding self-match)
        results = []
        for idx, sim in zip(I[0], D[0]):
            info = self.metadata[dim_key][idx]
            if info.model_name != model_name:  # Skip self-match
                results.append((info, float(sim)))
        
        return results[:k]

csw = CSWSearch()

# Example usage:
def add_params(model_list):
    
    # Add weight matrices from different models/layers

    for model_id in model_list:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        weights = model.state_dict()
        params = list(weights.keys())
        for param in params:
            csw.add_weight_matrix(model_id, param_name=param, weight_matrix=weights[param])
    
    # Search for similar weights to W1
    # similar_weights = csw.find_similar_weights("model1", W1)

    # return similar_weights

def get_similar_param(param, k=5):
    return csw.find_similar_weights("--", param, k=k)

def main():
    model_list = ["meta-llama/Llama-2-7b-hf", "lmsys/vicuna-7b-v1.5"]
    # yaml.load(open(args.models, 'r'), Loader=Loader)
    add_params(model_list)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    weights = model.state_dict()
    attn_name = "model.layers.0.self_attn.o_proj.weight"

    print(get_similar_param(weights[attn_name]))

    return

if __name__ == "__main__": 
    main()