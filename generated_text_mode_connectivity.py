import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from utils import interpolate_models

######################################################################################################

model_a_name = "meta-llama/Llama-2-7b-hf"

tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
pipeline_a = transformers.pipeline(
    "text-generation",
    model=model_a_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

######################################################################################################

model_b_name = "codellama/CodeLlama-7b-hf"
model_b_name = "lmsys/vicuna-7b-v1.1"

tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
pipeline_b = transformers.pipeline(
    "text-generation",
    model=model_b_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

######################################################################################################

start = ""

texts_a = []
texts_b = []

# generation_a = pipeline_a(start, max_new_tokens = 20, num_return_sequences=10, pad_token_id=tokenizer_a.eos_token_id)
# generation_b = pipeline_b(start, max_new_tokens = 20, num_return_sequences=10, pad_token_id=tokenizer_b.eos_token_id)

# for i in range(10):
#     texts_a.append(generation_a[i]['generated_text'])
#     texts_b.append(generation_b[i]['generated_text'])

for i in range(20):
    generation_a = pipeline_a(start, max_new_tokens = 20, num_return_sequences=1, pad_token_id=tokenizer_a.eos_token_id, do_sample=True)
    generation_b = pipeline_b(start, max_new_tokens = 20, num_return_sequences=1, pad_token_id=tokenizer_b.eos_token_id, do_sample=True)
    texts_a.append(generation_a[0]['generated_text'])
    texts_b.append(generation_b[0]['generated_text'])

pipeline_a.model.to("cpu")
pipeline_b.model.to("cpu")

print(texts_a)
print("######################################################################################################")
print(texts_b)
print("######################################################################################################")

del pipeline_a.model, pipeline_b.model
del pipeline_a, pipeline_b
torch.cuda.empty_cache()

######################################################################################################

model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.bfloat16)
model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.bfloat16)

######################################################################################################

losses = []

# alphas = [round(alpha * alpha_step, 2) for alpha in range(int(1/alpha_step + 1))]
# if endpoints == False:
#     alphas = alphas[1:-1]

alphas = [0, 0.5, 1]

for alpha in alphas:
    interpolated_model = interpolate_models(model_a, model_b, alpha).half().to('cuda')

    loss = []

    for text in texts_a:
        inputs = tokenizer_a(text, return_tensors = "pt").to('cuda')
        temp = interpolated_model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        loss.append(temp.item())
        inputs = inputs.to('cpu')
        del inputs
    
    for text in texts_b:
        inputs = tokenizer_b(text, return_tensors = "pt").to('cuda')
        temp = interpolated_model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        loss.append(temp.item())
        inputs = inputs.to('cpu')
        del inputs

    loss_mean = sum(loss) / len(loss)
    # perplexity = math.exp(loss_mean)

    # perplexities.append(perplexity)
    losses.append(loss_mean)

    interpolated_model.to("cpu")
    del interpolated_model, loss
    torch.cuda.empty_cache()

    print("alpha = " + str(alpha) + " | " + str(loss_mean))

