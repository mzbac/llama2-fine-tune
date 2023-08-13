import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_name ="./merged_peft/final_merged_checkpoint"
adapter_path = "./results/final_checkpoint"
# adapter_path = "./dpo_results/final_checkpoint"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer.encode("An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```here is how write for loop in js```\n### Output:", return_tensors="pt").to(DEV)

generate_kwargs = dict(
    input_ids=inputs,
    temperature=0.2, 
    top_p=0.95, 
    top_k=40,
    max_new_tokens=500,
    repetition_penalty=1.3
)
outputs = model.generate(**generate_kwargs)
print(tokenizer.decode(outputs[0]))