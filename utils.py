import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from scipy.optimize import linear_sum_assignment as LAP


def compute_jsd_stable(model_a, model_b, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_a = model_a(**inputs, labels=inputs["input_ids"])
        outputs_b = model_b(**inputs, labels=inputs["input_ids"])
        logits_a = outputs_a.logits.squeeze()
        logits_b = outputs_b.logits.squeeze()

        log_probs_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
        log_probs_b = torch.nn.functional.log_softmax(logits_b, dim=-1)

        m = 0.5 * (log_probs_a + log_probs_b)
        log_m = torch.logsumexp(m, dim=-1, keepdim=True)

        kl_div_a_m = (log_probs_a - log_m).sum(dim=-1)
        kl_div_b_m = (log_probs_b - log_m).sum(dim=-1)

        js_divergence = 0.5 * (kl_div_a_m + kl_div_b_m)

    return js_divergence.mean().item()


def compute_jsd(model_a, model_b, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_a = model_a(**inputs, labels=inputs["input_ids"])
        outputs_b = model_b(**inputs, labels=inputs["input_ids"])

    logits_a = outputs_a.logits.squeeze()
    logits_b = outputs_b.logits.squeeze()

    probs_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
    probs_b = torch.nn.functional.log_softmax(logits_b, dim=-1)

    m = 0.5 * (probs_a.exp() + probs_b.exp()).log()
    kl_div_a_m = torch.kl_div(probs_a, m, reduction="batchmean")
    kl_div_b_m = torch.kl_div(probs_b, m, reduction="batchmean")

    js_divergence = 0.5 * (kl_div_a_m + kl_div_b_m)

    return js_divergence.item()


def calculate_l2_distance(model1, model2):
    """
    Calculates the L2 distance between two PyTorch models with the same architecture.
    Args:
        model1 (AutoModelForCausalLM): The first model.
        model2 (AutoModelForCausalLM): The second model.
    Returns:
        float: The L2 distance between the two models.
    Raises:
        ValueError: If the parameter names or shapes do not match between the two models.
    """
    total_squared_diff = 0
    total_params = 0
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.shape != param2.shape:
            if name1 in ["model.embed_tokens.weight", "lm_head.weight"]:
                print(
                    f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )
        squared_diff = torch.sum((param1 - param2) ** 2)
        total_squared_diff += squared_diff.item()
        total_params += param1.numel()
    l2_distance = (total_squared_diff / total_params) ** 0.5
    return l2_distance


def calculate_perplexity(model, tokenizer, text, device):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        nll = loss.item()
        perplexity = torch.exp(torch.tensor(nll)).item()
    return perplexity


def interpolate_models(
    model_a, model_b, alpha=0.5, model_arch="meta-llama/Llama-2-7b-hf"
):
    """Linearly Interpolate between two model's parameters, conditioned on architecture."""
    interpolated_state_dict = {}
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    if model_arch == "meta-llama/Llama-2-7b-hf":
        vocab_size = 32000
    else:
        vocab_size = 32000
    for key in state_dict_a:
        # check that the vocabularies are the same length, if not
        # truncate to smaller vocab
        dim_a = state_dict_a[key].size(0)
        dim_b = state_dict_b[key].size(0)
        if key in ["model.embed_tokens.weight", "lm_head.weight"] and (
            dim_a != vocab_size or dim_b != vocab_size
        ):
            # hard code vocab to llama2 which is 32000
            interpolated_state_dict[key] = (1 - alpha) * state_dict_a[key][
                :vocab_size, :
            ] + alpha * state_dict_b[key][:vocab_size, :]
        else:
            interpolated_state_dict[key] = (1 - alpha) * state_dict_a[
                key
            ] + alpha * state_dict_b[key]

    model_interpolated = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", state_dict=interpolated_state_dict
    )
    return model_interpolated


def interpolate_models_truncate(model_a, model_b, alpha=0.5):
    """Interpolate between two model's parameters. Truncates to first 32000 rows of vocab."""
    interpolated_state_dict = {}
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    for key in state_dict_a:
        # check that the vocabularies are the same length, if not
        # truncate to smaller vocab
        dim_a = state_dict_a[key].size(0)
        dim_b = state_dict_b[key].size(0)
        if key in ["model.embed_tokens.weight", "lm_head.weight"] and (dim_a != dim_b):
            if dim_a < dim_b:
                interpolated_state_dict[key] = (1 - alpha) * state_dict_a[
                    key
                ] + alpha * state_dict_b[key][:32000, :]
            else:
                interpolated_state_dict[key] = (1 - alpha) * state_dict_a[key][
                    :32000, :
                ] + alpha * state_dict_b[key]
        else:
            interpolated_state_dict[key] = (1 - alpha) * state_dict_a[
                key
            ] + alpha * state_dict_b[key]

    model_interpolated = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        state_dict=interpolated_state_dict,
        torch_dtype=torch.bfloat16,
    )
    return model_interpolated.to(torch.bfloat16)

def compute_emb_lap(model_a,model_b):
  perm = match_emb(model_a,model_b)

  return spcor(perm,torch.arange(len(perm)))

def match_mlp(base_model,ft_model,i=0):
  base_wmat = base_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight']
  ft_wmat = ft_model.state_dict()['model.layers.'+str(i)+'.mlp.gate_proj.weight']

  perm = match_wmats(base_wmat,ft_wmat)

  return perm

def match_emb(base_model,ft_model,layer='inp'):
  if layer == 'inp':
    weight_id = 'model.embed_tokens.weight'
  if layer == 'out':
    weight_id = 'lm_head.weight'

  base_wmat = base_model.state_dict()[weight_id].T
  ft_wmat = ft_model.state_dict()[weight_id].T

  perm = match_wmats(base_wmat,ft_wmat)
  return perm

def match_wmats(wmat0,wmat1):
  dists = pdists(wmat0,wmat1).type(torch.float64)
  perm = LAP(dists)[1]

  return perm # wmat1[perm] should match wmat0

def pdists(x,y):
  with torch.no_grad():
    xsum = torch.sum(torch.square(x),axis=-1)
    ysum = torch.sum(torch.square(y),axis=-1)

    dists = xsum.view(-1,1) + ysum.view(1,-1) - 2 * x@y.T

  return dists
  
def spcor(x,y):
  n = len(x)
  return 1 - torch.sum(6 * torch.square(x-y)) / (n * (n - 1))
