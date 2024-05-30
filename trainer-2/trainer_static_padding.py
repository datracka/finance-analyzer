import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
import numpy as np
import evaluate
import torch


def tokenize_function(dataset):
    return tokenizer(dataset["statement"], max_length=128, padding="max_length", truncation=True)

## Prepare dataset from CSV https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt

# Read data from CSV
data = pd.read_csv("csv/bank_statements_ing.csv")

# Map categories to integers
category_mapping = {
    'Ocio y restauracion': 1, 'Alimentacion': 2, 'Compras': 3, 'Reintegros': 4,  # Add more categories as needed
    'Seguros & medicos': 5, 'Ingresos': 6, 'Suscripciones': 7, 'Descartado': 8,
    'Vehiculo y transporte': 9, 'Alquiler': 10, 'Company': 11, 'Limpieza & Babysitter': 12,
    'Pisos / Inversiones': 13, 'ONGs': 14, 'Educacion': 15,
    'Luz & Electricidad & Basuras': 16, 'Tasas, comisiones e impuestos': 17
}

data['amount'] = data['amount'].replace(',', '', regex=True).astype(float)
data['label'] = data['category'].map(category_mapping)

# Optionally drop unwanted columns
data = data.drop(['date', 'category', 'amount'], axis=1)

features = Features({
    'idx': Value(dtype='int32'),
    'label': ClassLabel(num_classes=17, names=list(category_mapping.keys())),
    'statement': Value(dtype='string'),
})

# Create Hugging Face dataset from DataFrame
hf_dataset = Dataset.from_pandas(data, features=features)

# Split dataset into train, validation, and test
dataset_train = hf_dataset.train_test_split(test_size=0.2)
dataset_validation_test = dataset_train['test'].train_test_split(test_size=0.5)
dataset_dict = DatasetDict({
    'train': dataset_train['train'],
    'validation': dataset_validation_test['train'],
    'test': dataset_validation_test['test']
})

checkpoint = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokenize the dataset
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# log: check the tokenizer length of a sample from our datasets
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "statement"]}
print([len(x) for x in samples["input_ids"]])

## Model Fine Tunning https://huggingface.co/docs/transformers/training

output_folder = "finance_trainer"

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=17)

training_args = TrainingArguments(
    output_dir=output_folder, 
    eval_strategy="epoch",
    per_device_train_batch_size=8,  # You might want to explicitly define batch sizes
    per_device_eval_batch_size=8)

metric = evaluate.load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# trainer.train()

# Save the model
# trainer.save_model()

# a save tokenizer in the same directory. It stores all the preprocessing done with the data
# tokenizer.save_pretrained(output_folder)