import os
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

login(token=HUGGINGFACE_TOKEN)

device = "cuda"
dtype = torch.bfloat16

def init():
    global tokenizer, model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/model"
    )
    tokenizer_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/tokenizer"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
    logging.info("Init complete")

def moderate(chat):
    global tokenizer, model
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def run(data):
    logging.info("Request received")
    inputs = json.loads(data)
    conversation = inputs['conversation']
    logging.info("Request processed")
    return moderate(conversation)











