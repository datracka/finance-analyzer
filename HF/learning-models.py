from transformers import AutoTokenizer
import json

### learning tokenization

sequence = "Using a Transformer network is simple"

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=True)
print("### get tokens ##")
print(tokenizer.tokenize(sequence))
print("\n")

print("### convert tokens to ids ##")
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)))
print("\n")

print("## ids to special characters ##")
print(tokenizer.prepare_for_model(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)))["input_ids"])
print("\n")

print("### get decoded string ##")
print(tokenizer.decode(tokenizer.prepare_for_model(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)))["input_ids"]))
print("\n")

print("## overall process ##")
inputs = tokenizer(sequence)
print(inputs)
print("\n")
# tokenizer.save_pretrained('./model')
