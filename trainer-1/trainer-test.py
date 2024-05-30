
from datasets import Dataset, DatasetDict
import pandas as pd

# Load your CSV
df = pd.read_csv('csv/bank_statements.csv')

# Convert DataFrame to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Split the dataset
train_testvalid = dataset.train_test_split(test_size=0.2)
valid_test = train_testvalid['test'].train_test_split(test_size=0.5)

## create the mapping label / integer

label_dict = {
    0: "label_one",
    1: "label_two",
    2: "Alimentación",
    3: "Compras",
    4: "",
    5: "",
    6: "Ocio y restauracion",
    7: "",
    8: "",
    9: "seguros & médicos",
    10: "Transferencias"

    # Add more labels as needed
}

dataset_dict = DatasetDict({
    'train': train_testvalid['train'],
    'validation': valid_test['train'],
    'test': valid_test['test']
})
train_dataset = dataset_dict["train"]
print(train_dataset.features)