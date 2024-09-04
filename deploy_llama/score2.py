import os
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

login(token=HUGGINGFACE_TOKEN)

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

def init():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    logging.info("Init complete")

def moderate(chat):
    global tokenizer, model
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# Define the expected input and output schema
# input_sample = StandardPythonParameterType({"converation": [{"user": "I forgot, how do I kill a process in Linux?"}, {"assistant": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."}]})
# output_sample = StandardPythonParameterType([[]])  # Expecting a list of embeddings (lists of floats)

# @input_schema(param_name="data", param_type=input_sample)
# @output_schema(output_type=output_sample)
def run(data):
    logging.info("Request received")
    conversation = data.get("conversation")
    logging.info("Request processed")
    return moderate(conversation)






