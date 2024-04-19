# run the project

> pipenv shell

to start the virtual environment

> pipenv install *

Virtual environment is managed previouly by pyenv https://realpython.com/intro-to-pyenv/

Folder `csv` with dataset is ignored by now.

## Train a model

* `python trainer` to tune a pretrained model. Pretrained model is `bert-base-multilingual-uncased`
  * created model store in `/model` folder (gitignored)
* `python predictor`  and `python pipeline` to test model works
  
## to install package

> pipenv install <whatever>

## Extra Ollama: Create modelfile (not used in this demo)

ollama create finance_analyzer_llama2 -f ./finance_analyer.txt
ollama run finance_analyzer_llama2


## TODO

- x] You need to create your own dataset Check Final Version from-chat-gpt.py
  - [x] Create Dataset
  - [x] Check if DataSet is balanced
  - [x] Create category labels
- [x] Model Created
- [-] Model tested

ERROR: There is an error when running predictor.py with a sample statement to test

```
  File "/Users/vicensfayos/.local/share/virtualenvs/finance-zruwXuID/lib/python3.12/site-packages/torch/nn/functional.py", line 2237, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: index out of range in self
```

-[ ] Learn everything about fintuning a model and do a hello world test read: https://huggingface.co/learn

I am here https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
