import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import csv
import os
import time
import gc
import scipy
from scipy.spatial import distance

from datasets import load_dataset
from permutation_tests import group_texts

def compute_jsd_tokenized(model_a, model_b, input_ids, attention_mask, labels):
    with torch.no_grad():
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

        probs_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
        probs_b = torch.nn.functional.log_softmax(logits_b, dim=-1)

        probs_a = probs_a[:, :32000]
        probs_b = probs_b[:, :32000]

        softmax_a = torch.softmax(logits_a, dim=-1)
        softmax_b = torch.softmax(logits_b, dim=-1)

        softmax_a = softmax_a[:, :32000]
        softmax_b = softmax_b[:, :32000]
        print(softmax_a.shape)

        M = 0.5 * (softmax_a + softmax_b)

        jsd = 0.5 * (F.kl_div(M.log(), softmax_a) +
                   F.kl_div(M.log(), softmax_b))

        return jsd.item()

def compute_jsd_text(model_a, model_b, text):
    tokenizer_a = AutoTokenizer.from_pretrained(model_a.config._name_or_path)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b.config._name_or_path)

    inputs_a = tokenizer_a(text, return_tensors = "pt").to('cuda')
    outputs_a = model_a(input_ids = inputs_a["input_ids"])

    inputs_b = tokenizer_b(text, return_tensors = "pt").to('cuda')
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

def main():
    # update file based on current working directory
    filepath = f"{os.getcwd()}/model_pairs_jsd.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_list = [
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-7b-hf",
            "openlm-research/open_llama_7b",
            "huggyllama/llama-7b",
            "lmsys/vicuna-7b-v1.5",
            "EleutherAI/llemma_7b",
            "lmsys/vicuna-7b-v1.1",
            "microsoft/Orca-2-7b",
            "LLM360/Amber",
        ]
    # init datasets before hand to prefetch data
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    eval_dataset = load_dataset("dlwh/wikitext_103_detokenized", split="test")
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = eval_dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=1,
        do_eval=True,
        report_to=None,
        dataloader_num_workers=4,
        use_cpu=True,

    )
    # hack to get dataloader from HF. For some reason simply iterating
    # through dataloader is faster than calling evaluate()
    # but Trainer won't return dataloader without model
    model_temp = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
    trainer = Trainer(model=model_temp, args=training_args, eval_dataset=lm_datasets)
    eval_dataloader = trainer.get_test_dataloader(lm_datasets)
    
    # reclaim RAM memory
    del model_temp
    torch.cuda.empty_cache()
    gc.collect()
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            # if(i == 0 and j <= 6): continue 
            print("Starting...")
            time0 = time.time()

            model_a_name = model_list[i]
            model_b_name = model_list[j]

            print(model_a_name, model_b_name)

            model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
            model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

            # move to cuda
            model_a = model_a.to(device)
            model_b = model_b.to(device)

            # for batch in eval_dataloader:
            #     input_ids = batch["input_ids"].to(device)
            #     attention_mask = batch["attention_mask"].to(device)
            #     labels = batch["labels"].to(device)

            #     jsd1, jsd2 = compute_jsd_tokenized(model_a, model_b, input_ids, attention_mask, labels)
            #     print(jsd1, jsd2)

                
            #     break

            text = 'This is a test message; we use this message to calculate the parameter shift'

            jsd = compute_jsd_text(model_a, model_b, text)
            
            # input_ids = input_ids.to('cpu')
            # attention_mask = attention_mask.to('cpu')
            # labels = labels.to('cpu')
            model_a = model_a.to('cpu')
            model_b = model_b.to('cpu')
            del model_a, model_b
            # del input_ids, attention_mask, labels

            torch.cuda.empty_cache() 
            gc.collect()
            

            csv_header = ["Model Pair", "jsd"]

            if not os.path.exists(filepath):
                with open(filepath, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)
            
            with open(filepath, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                model_pair = f"{model_a_name} vs {model_b_name}"
                row = [model_pair, jsd]
                writer.writerow(row)
            
            print("done! time was")
            print(str(time.time() - time0))

if __name__ == "__main__":
    main()
