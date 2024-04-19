from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./model')

model = AutoModelForSequenceClassification.from_pretrained('./model')

# The new statement you want to classify
statement = "Pago en PIZZERIA EATMOSTERA BARCELONA ES"

# Preprocess the statement
inputs = tokenizer(statement, truncation=True, padding=True, return_tensors="pt")

# Get the model's predictions
outputs = model(**inputs)

# The predictions are logits (unnormalized scores) for each category
# Use the softmax function to convert these into probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the category with the highest probability
predicted_category = torch.argmax(probabilities).item()
