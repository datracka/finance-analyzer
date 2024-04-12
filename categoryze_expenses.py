# from langchain_community.llms import Ollama
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

load_dotenv()

# llm = Ollama(model="llama2")
# llm.invoke("In japan, how many people speak english?")


df = pd.read_csv('csv/1_year_bank_statements_ING_100_only.csv')
data_dict = df.to_dict()
dataset = Dataset.from_dict(data_dict)

# Split the dataset into train and test
train_dataset = dataset.train_test_split(test_size=0.2)['train']
test_dataset = dataset.train_test_split(test_size=0.2)['test']