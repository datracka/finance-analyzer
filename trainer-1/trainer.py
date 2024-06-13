""" 

Dataset Splitting

Split your dataset into at least two subsets: training and validation. 
A common split ratio is 80% for training and 20% for validation. 
If your dataset is large enough, consider setting aside a portion as a 
test set (e.g., 70% training, 15% validation, 15% test).

Encoding Categories

Since your model will output categories, you need to convert 
the category labels from text to a format that the model can process:

Label encoding: Convert each category name into a unique integer. 
Maintain a mapping dictionary to convert back to category names 
from predictions.

 """
## Loading Data into Hugging Face

from datasets import Dataset, DatasetDict
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load your CSV
df = pd.read_csv('csv/bank_statements.csv')

# Convert DataFrame to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Split the dataset
train_testvalid = dataset.train_test_split(test_size=0.2)
valid_test = train_testvalid['test'].train_test_split(test_size=0.5)
dataset_dict = DatasetDict({
    'train': train_testvalid['train'],
    'validation': valid_test['train'],
    'test': valid_test['test']
})

# Processing function to tokenize text


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=True)
tokenizer.save_pretrained('./model')

def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples['Statements'], padding="max_length", truncation=True)
    # Add the labels
    tokenized['labels'] = examples['Label']
    return tokenized

# Apply tokenization
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=df['Label'].nunique())

training_args = TrainingArguments(
    output_dir='./model',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()
# Save the model
trainer.save_model()