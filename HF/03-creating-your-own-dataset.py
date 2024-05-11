from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset.features)