from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

llm = Ollama(model="llama2")
llm.invoke("In japan, how many people speak english?")

import pandas as pd
df = pd.read_csv('expenses_test.csv')

