from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "models_created/finance_trainer"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

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
predicted_category_id = torch.argmax(probabilities).item()

label_names = [
            "Ocio y restauracion",
            "Alimentacion",
            "Compras",
            "Reintegros",
            "Seguros & medicos",
            "Ingresos",
            "Suscripciones",
            "Descartado",
            "Vehiculo y transporte",
            "Alquiler",
            "Company",
            "Limpieza & Babysitter",
            "Pisos / Inversiones",
            "ONGs",
            "Educacion",
            "Luz & Electricidad & Basuras",
            "Tasas, comisiones e impuestos",
    ]  

print(predicted_category_id)
print(label_names[predicted_category_id])
