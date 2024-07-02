import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tracing.utils.evaluate import evaluate
from tracing.utils.evaluate import prepare_hf_dataset,prepare_hf_dataloader, prepare_programming_dataset, load_generated_datasets


def statistic(base_model, ft_model, dataloader, device="cuda"):
    return compute_jsd(base_model, ft_model, dataloader, device)

def statistic_stable(base_model, ft_model, dataloader, device="cuda"):
    return compute_jsd_stable(base_model, ft_model, dataloader, device)

def compute_jsd(base_model, ft_model, dataloader, device="cuda"):
    jsds = []

    base_model.to(device)
    ft_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_base = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs_ft = ft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits_base = outputs_base.logits.squeeze()
            logits_ft = outputs_ft.logits.squeeze()

            softmax_base = torch.softmax(logits_base, dim=-1)
            softmax_ft = torch.softmax(logits_ft, dim=-1)

            # Truncate the softmax outputs to the first 32000 dimensions
            softmax_base = softmax_base[:, :32000]
            softmax_ft = softmax_ft[:, :32000]

            m = 0.5 * (softmax_base + softmax_ft)
            jsd = 0.5 * (F.kl_div(m.log(), softmax_base) +
                         F.kl_div(m.log(), softmax_ft))

            jsds.append(jsd.item())

    base_model.to("cpu")
    ft_model.to("cpu")
    print(sum(jsds))
    return sum(jsds)

def compute_jsd_stable(base_model, ft_model, dataloader, device="cuda"):
    jsds = []

    base_model.to(device)
    ft_model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_base = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs_ft = ft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits_base = outputs_base.logits.squeeze()
            logits_ft = outputs_ft.logits.squeeze()

            # Determine the minimum vocabulary size between the two models
            min_vocab_size = min(logits_base.size(-1), logits_ft.size(-1))

            # Truncate the logits to the minimum vocabulary size
            logits_base = logits_base[..., :min_vocab_size]
            logits_ft = logits_ft[..., :min_vocab_size]

            log_probs_base = F.log_softmax(logits_base, dim=-1)
            log_probs_ft = F.log_softmax(logits_ft, dim=-1)

            m = 0.5 * (log_probs_base.exp() + log_probs_ft.exp())
            log_m = m.log()

            kl_div_base_m = (log_probs_base - log_m).sum(dim=-1)
            kl_div_ft_m = (log_probs_ft - log_m).sum(dim=-1)

            jsd = 0.5 * (kl_div_base_m + kl_div_ft_m).mean()
            jsds.append(jsd.item())

    base_model.to("cpu")
    ft_model.to("cpu")

    return sum(jsds)


if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf" # 'openlm-research/open_llama_7b' # 'lmsys/vicuna-7b-v1.5' 
    ft_model_name = "codellama/CodeLlama-7b-hf" # 'openlm-research/open_llama_7b_v2' # 'LLM360/Amber' # "lmsys/vicuna-7b-v1.1"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    dataset = load_generated_datasets(base_model_name, ft_model_name, 512, base_tokenizer, ["text"])
    dataloader = prepare_hf_dataloader(dataset, 1)

    print(statistic(base_model, ft_model, dataloader))