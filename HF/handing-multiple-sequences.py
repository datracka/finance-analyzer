from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', use_fast=True)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
 