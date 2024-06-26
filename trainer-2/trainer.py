import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
import numpy as np
import evaluate

def tokenize_function(dataset):
    return tokenizer(dataset["statement"], truncation=True, padding=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


## configuration

csv_file="csv/bank_statements_ing_subset.csv"
checkpoint = "PlanTL-GOB-ES/roberta-base-bne"
output_folder="finance-trainer_oberta-base-bne"

# Map categories to integers
num_categories=8
category_mapping = {
    'Ocio y restauracion': 0, 'Alimentacion': 1, 'Compras': 2, 'Reintegros': 3,  # Add more categories as needed
    'Suscripciones': 4, 'ONGs': 5, 'Tasas, comisiones e impuestos': 6, 'Descartado': 7
}

#    'Vehiculo y transporte': 8, 'Alquiler': 9, 'Company': 10, 'Limpieza & Babysitter': 11,
#    'Pisos / Inversiones': 12, 'Ingresos': 13, 'Educacion': 14,
#    'Luz & Electricidad & Basuras': 15, 'Seguros & medicos': 16

## end configuration



## Prepare dataset from CSV https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt

# Read data from CSV
data = pd.read_csv(csv_file)

data['amount'] = data['amount'].replace(',', '', regex=True).astype(float)

# check if there are any categories that are not in the mapping
# print(data[~data['category'].isin(category_mapping.keys())])

data['label'] = data['category'].map(category_mapping).astype(int)

# Optionally drop unwanted columns
data = data.drop(['date', 'category', 'amount'], axis=1)

features = Features({
    'idx': Value(dtype='int32'),
    'label': ClassLabel(num_classes=num_categories, names=list(category_mapping.keys())),
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

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokenize the dataset
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# add dynamic padding to the longest sequence
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# log: check the tokenizer length of a sample from our datasets
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "statement"]}
# print([len(x) for x in samples["input_ids"]])

# log: check if the batch is padding dinamically to the longest sequence
# batch = data_collator(samples)
# print({k: v.shape for k, v in batch.items()})


## Model Fine Tunning https://huggingface.co/docs/transformers/training

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_categories)

training_args = TrainingArguments(
    output_dir=output_folder, 
    eval_strategy="epoch",
    per_device_train_batch_size=8,  # You might want to explicitly define batch sizes
    per_device_eval_batch_size=8)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()

# Save the model
trainer.save_model()

# a save tokenizer in the same directory. It stores all the preprocessing done with the data
tokenizer.save_pretrained(output_folder)
