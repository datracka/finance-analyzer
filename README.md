# run the project

> pipenv shell

to start the virtual environment

Virtual environment is managed previouly by pyenv https://realpython.com/intro-to-pyenv/

## Create modelfile

ollama create finance_analyzer_llama2 -f ./finance_analyer.txt
ollama run finance_analyzer_llama2

## to install package

> pipenv install <whatever>

## next steps

- [x] You need to create your own dataset: https://huggingface.co/learn/nlp-course/chapter5/5

- [ ] use it to train https://huggingface.co/docs/transformers/training

- Pending, to choose the proper Pretrained model maybe this https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli?
  
- Understand, how they connect to my local model Ollama.

