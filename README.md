# run the project

> pipenv shell

to start the virtual environment

Virtual environment is managed previouly by pyenv https://realpython.com/intro-to-pyenv/

Folder `csv` with dataset is ignored by now.

## Create modelfile

ollama create finance_analyzer_llama2 -f ./finance_analyer.txt
ollama run finance_analyzer_llama2

## to install package

> pipenv install <whatever>

## TODO

- x] You need to create your own dataset Check Final Version from-chat-gpt.py
  - [x] Create Dataset
  - [x] Check if DataSet is balanced
  - [x] Create category labels
- [x] Model Created
- [-] Model tested

ERROR: There is an error when running predictor.py with a sample statement to test

-[ ] Learn everything about fintuning a model and do a hello world test read: https://huggingface.co/learn

I am here https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
