from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login
from langchain_community.embeddings import LlamaCppEmbeddings
from datetime import datetime

def create_bnb_config(load_in_4bit, bnb_4_bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
        Configure model quantization using bitsandbytes to speed up training and inference
        :param load_in_4bit: Load the model in 4-bit precision mode
        :param bnb_4_bit_use_double_quant: nested quantization for 4-bit model
        :param bnb_4bit_quant_type: The quantization type for 4-bit model
        :param bnb_4bit_compute_dtype: The compute dtype for 4-bit model
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4_bit_use_double_quant=bnb_4_bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    return bnb_config

def load_model(model_name, bnb_config, max_memory=22960):
    """
        Load the model and tokenizer
        :param model_name: Huggingface model name

    """
    n_gpus = torch.cuda.device_count()
    max_memory = f'{max_memory}MB'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 quantization_config=bnb_config,
                                                 max_memory= {i: max_memory for i in range(n_gpus)})
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def mean_pooling(embeddings, attention_mask):
    """
    Pool the embeddings by averaging the token embeddings, weighted by the attention mask.
    """
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(text, model, tokenizer):
    """
    Generate embeddings for a given text using the fine-tuned model.
    """
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors='pt')

    # Use the model to get the embeddings
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    # Extract the embeddings (e.g., from the last hidden state)
    embeddings = outputs.hidden_states[-1]
    
    pooled_embeddings = mean_pooling(embeddings, tokens['attention_mask'])
    return pooled_embeddings

if __name__ == "__main__":
    model_name = "XiWangEric/Utterance_emotion_llama2_7b"
    #Bitsandbytes parameters
    # Activate 4-bit precision base model loading
    load_in_4bit = True

    # Activate nested quantization for 4-bit model 
    bnb_4bit_use_double_quant = True

    # The quantization type for 4-bit model (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # The compute dtype for 4-bit model
    bnb_4bit_compute_dtype = torch.bfloat16

    bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
    model, tokenizer = load_model(model_name, bnb_config)

    # Example text for generating embeddings
    text = "Sample input text to generate embeddings"
    embeddings = get_embeddings(text, model, tokenizer)
    
    print(embeddings.shape)
    print(embeddings)