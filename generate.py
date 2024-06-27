import torch
import transformers
from transformers import AutoTokenizer
import json
import yaml
from yaml import Loader

def generate_texts(model_name, tokenizer, num, max_tokens, device="cuda"):
  pipeline = transformers.pipeline(
      "text-generation",
      model=model_name,
      torch_dtype=torch.float16,
      trust_remote_code=True,
      device_map=device
  )

  texts = []

  start = ""
  for i in range(num):
    print(i)
    generation = pipeline(start, max_new_tokens = max_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    texts.append(generation[0]['generated_text'])

  pipeline.model.to("cpu")
  del pipeline.model, pipeline
  torch.cuda.empty_cache()

  return texts

def save_texts(texts, filename):
  text = ""
  for t in texts:
    text += t
    text += " "

  dictionary = {"text": text}
  json_object = json.dumps(dictionary)

  with open(filename, "w") as outfile:
     outfile.write(json_object)

def main():

    # model_paths = yaml.load(open("model-tracing/config/model_list.yaml", 'r'), Loader=Loader)
    # base_models = model_paths["base_models"]
    # ft_models = model_paths["ft_models"]

    base_models = ["meta-llama/Llama-2-7b-hf"]
    ft_models = ["codellama/CodeLlama-7b-hf"]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    for model_name in base_models: 
        print(model_name)
        
        texts = generate_texts(model_name, tokenizer, 10, 2048)
        file_name = "/juice4/scr4/nlp/model-tracing/generations/long/" + model_name.replace("/","-") + "_gentext.json"
        save_texts(texts, file_name)

    for model_name in ft_models: 
        print(model_name)
        
        texts = generate_texts(model_name, tokenizer, 10, 2048)
        file_name = "/juice4/scr4/nlp/model-tracing/generations/long/" + model_name.replace("/","-") + "_gentext.json"
        save_texts(texts, file_name)

if __name__ == "__main__":
    main()