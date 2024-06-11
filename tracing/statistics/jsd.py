import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# From text, i.e. text = 'This is a test message; we use this message to calculate the parameter shift'

def compute_jsd_from_text(model_a, model_b, text, device="cuda"):
    tokenizer_a = AutoTokenizer.from_pretrained(model_a.config._name_or_path)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b.config._name_or_path)

    inputs_a = tokenizer_a(text, return_tensors = "pt").to(device)
    outputs_a = model_a(input_ids = inputs_a["input_ids"])

    inputs_b = tokenizer_b(text, return_tensors = "pt").to(device)
    outputs_b = model_b(input_ids = inputs_b["input_ids"])

    logits_a = outputs_a.logits.squeeze()
    logits_b = outputs_b.logits.squeeze()

    softmax_a = torch.softmax(logits_a, dim=-1)
    softmax_b = torch.softmax(logits_b, dim=-1)

    softmax_a = softmax_a[:, :32000]
    softmax_b = softmax_b[:, :32000]

    M = 0.5 * (softmax_a + softmax_b)

    jsd = 0.5 * (F.kl_div(M.log(), softmax_a) +
                F.kl_div(M.log(), softmax_b))

    return jsd.item()

def compute_jsd_from_dataloader(model_a, model_b, dataloader, device="cuda"):

    jsds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_a = model_a(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs_b = model_b(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits_a = outputs_a.logits.squeeze()
            logits_b = outputs_b.logits.squeeze()

            softmax_a = torch.softmax(logits_a, dim=-1)
            softmax_b = torch.softmax(logits_b, dim=-1)

            softmax_a = softmax_a[:, :32000]
            softmax_b = softmax_b[:, :32000]

            M = 0.5 * (softmax_a + softmax_b)

            jsd = 0.5 * (F.kl_div(M.log(), softmax_a) +
                        F.kl_div(M.log(), softmax_b))

            jsds.append(jsd)

    return jsds

def compute_jsd_stable(model_a, model_b, tokenizer, text, device="cuda"):
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


def compute_jsd(model_a, model_b, tokenizer, text, device="cuda"):
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
