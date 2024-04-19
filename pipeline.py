from transformers import pipeline

# Path to your fine-tuned model
model_path = './model'

# Create a pipeline with your model
classifier = pipeline('text-classification', model=model_path)

# Now you can use the pipeline to make predictions
result = classifier("Pago en PIZZERIA EATMOSTERA BARCELONA ES")

print(result)