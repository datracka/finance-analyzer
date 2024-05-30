# Finance Analyzer

Side project to learn about ML / AI, Model Fine tuning, PyTorch, Tensor Flow and Hugging Face libraries

## run the project

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

## TODOs

* [ ] (Optional) DeepLearning in https://course.fast.ai/ or https://www.deeplearning.ai/
* [ ] Learn how to use using Hugging face for loading a ollama model and learn about GGUF format
* [ ] improve accuracy of the model
* [ ] deploy it