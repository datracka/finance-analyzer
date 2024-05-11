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

- [x] You need to create your own dataset Check Final Version from-chat-gpt.py
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
TODOs

* [ ] (Optional) DeepLearning in https://course.fast.ai/ or https://www.deeplearning.ai/
* [x] Transformer models
  * [x] learn NPL https://huggingface.co/learn/nlp-course/chapter1/1
* [x] Using Transformers
  * [x] https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt
* [-] Fine Tuning a pretrained model https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt
  * [ ] DataSet labels are wrong https://docs.google.com/spreadsheets/d/16cdQz4u2L-ZUzBUueqNFgfrdi5KCkGqli16jUEg2WHI/edit?usp=sharing
  * [ ] Create DataSet ClassLabels (ClassLabel(num_classes=2, names) from the values I have as labels Value(dtype='int64', id=None) using `trainer-test.py`
  * [ ] https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt
* [ ] Learn how to use using Hugging face for loading a ollama model and learn about GGUF format 