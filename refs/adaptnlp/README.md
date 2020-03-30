<p align="center">
    <a href="https://github.com/Novetta/adaptnlp"> <img src="https://raw.githubusercontent.com/novetta/adaptnlp/master/docs/img/NovettaAdaptNLPlogo-400px.png" width="400"/></a>
</p>

<p align="center">
<strong> A high level framework and library for running, training, and deploying state-of-the-art Natural Language Processing (NLP) models for end to end tasks.</strong>
</p>
<p align="center">
    <a href="https://circleci.com/gh/Novetta/adaptnlp">
        <img src="https://img.shields.io/circleci/build/github/Novetta/adaptnlp/master">
    </a>
    <a href="https://badge.fury.io/py/adaptnlp">
        <img src="https://badge.fury.io/py/adaptnlp.svg">
    </a>
    <a href="https://github.com/Novetta/adaptnlp/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/novetta/adaptnlp">
    </a>
</p>


AdaptNLP allows users ranging from beginner python coders to experienced machine learning engineers to leverage
state-of-the-art NLP models and training techniques in one easy-to-use python package.

Built atop Zalando Research's Flair and Hugging Face's Transformers library, AdaptNLP provides Machine
Learning Researchers and Scientists a modular and **adaptive** approach to a variety of NLP tasks with an
**Easy** API for training, inference, and deploying NLP-based microservices.

**Key Features**

  - **[Full Guides and API Documentation](https://novetta.github.io/adaptnlp)**
  - [Tutorial](https://github.com/Novetta/adaptnlp/tree/master/tutorials) Jupyter/Google Colab Notebooks
  - Unified API for NLP Tasks with SOTA Pretrained Models (Adaptable with Flair and Transformer's Models)
    - Token Tagging 
    - Sequence Classification
    - Embeddings
    - Question Answering
    - <em> More in development </em>
  - Training and Fine-tuning Interface
    - Jeremy's **[ULM-FIT](https://arxiv.org/abs/1801.06146)** approach for transfer learning in NLP
    -  Fine-tuning Transformer's language models and task-specific predictive heads like Flair's `SequenceClassifier`
  - [Rapid NLP Model Deployment](https://github.com/Novetta/adaptnlp/tree/master/rest) with Sebastián's [FastAPI](https://github.com/tiangolo/fastapi) Framework
    - Containerized FastAPI app
    - Immediately deploy any custom trained Flair or AdaptNLP model
  - [Dockerizing AdaptNLP with GPUs](https://hub.docker.com/r/achangnovetta/adaptnlp)
    - Easily build and run AdaptNLP containers leveraging NVIDIA GPUs with Docker


## Quick Start

#### Requirements and Installation

##### Virtual Environment
To avoid dependency clustering and issues, it would be wise to install AdaptNLP in a virtual environment.
To create a new python 3.6+ virtual environment, run this command and then activate it however your operating
system specifies:

```
python -m venv venv-adaptnlp
```

##### AdaptNLP Install

Install using pip in your virtual environment:
```
pip install adaptnlp
```


#### Examples and General Use

Once you have installed AdaptNLP, here are a few examples of what you can run with AdaptNLP modules:

##### Named Entity Recognition with `EasyTokenTagger`

```python
from adaptnlp import EasyTokenTagger

## Example Text
example_text = "Novetta's headquarters is located in Mclean, Virginia."

## Load the token tagger module and tag text with the NER model 
tagger = EasyTokenTagger()
sentences = tagger.tag_text(text=example_text, model_name_or_path="ner")

## Output tagged token span results in Flair's Sentence object model
for sentence in sentences:
    for entity in sentence.get_spans("ner"):
        print(entity)

```

##### English Sentiment Classifier `EasySequenceClassifier`

```python
from adaptnlp import EasySequenceClassifier 

## Example Text
example_text = "Novetta is a great company that was chosen as one of top 50 great places to work!"

## Load the sequence classifier module and classify sequence of text with the english sentiment model 
classifier = EasySequenceClassifier()
sentences = classifier.tag_text(text=example_text, mini_batch_size=1, model_name_or_path="en-sentiment")

## Output labeled text results in Flair's Sentence object model
for sentence in sentences:
    print(sentence.labels)

```

##### Span-based Question Answering `EasyQuestionAnswering`

```python
from adaptnlp import EasyQuestionAnswering 

## Example Query and Context 
query = "What is the meaning of life?"
context = "Machine Learning is the meaning of life."
top_n = 5

## Load the QA module and run inference on results 
qa = EasyQuestionAnswering()
best_answer, best_n_answers = qa.predict_qa(query=query, context=context, n_best_size=top_n, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")

## Output top answer as well as top 5 answers
print(best_answer)
print(best_n_answers)
```

##### Sequence Classification Training `SequenceClassifier`
```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer 

# Specify corpus data directory and model output directory
corpus = "Path/to/data/directory" 
OUTPUT_DIR = "Path/to/output/directory" 

# Instantiate AdaptNLP easy document embeddings module, which can take in a variable number of embeddings to make `Stacked Embeddings`.  
# You may also use custom Transformers LM models by specifying the path the the language model
doc_embeddings = EasyDocumentEmbeddings(model_name_or_path="bert-base-cased", methods = ["rnn"])

# Instantiate Sequence Classifier Trainer by loading in the data, data column map, and embeddings as an encoder
sc_trainer = SequenceClassifierTrainer(corpus=corpus, encoder=doc_embeddings, column_name_map={0: "text", 1:"label"})

# Find Learning Rate
learning_rate = sc_trainer.find_learning_rate(output_dir=OUTPUT_DIR)

# Train Using Flair's Sequence Classification Head
sc_trainer.train(output_dir=OUTPUT_DIR, learning_rate=learning_rate, max_epochs=150)


# Predict text labels with the trained model using `EasySequenceClassifier`
from adaptnlp import EasySequenceClassifier
example_text = '''Where was the Queen's wedding held? '''
classifier = EasySequenceClassifier()
sentences = classifier.tag_text(example_text, model_name_or_path=OUTPUT_DIR / "final-model.pt")
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

##### Transformers Language Model Fine Tuning `LMFineTuner`

```python
from adaptnlp import LMFineTuner

# Specify Text Data File Paths
train_data_file = "Path/to/train.csv"
eval_data_file = "Path/to/test.csv"

# Instantiate Finetuner with Desired Language Model
finetuner = LMFineTuner(train_data_file=train_data_file, eval_data_file=eval_data_file, model_type="bert", model_name_or_path="bert-base-cased")
finetuner.freeze()

# Find Optimal Learning Rate
learning_rate = finetuner.find_learning_rate(base_path="Path/to/base/directory")
finetuner.freeze()

# Train and Save Fine Tuned Language Models
finetuner.train_one_cycle(output_dir="Path/to/output/directory", learning_rate=learning_rate)

```

## Tutorials

Look in the [Tutorials](tutorials) directory for a quick introduction to the library and its very simple
and straight forward use cases:

  1. [Token Classification: NER, POS, Chunk, and Frame Tagging](tutorials/1.%20Token%20Classification)
      - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/1.%20Token%20Classification/token_tagging.ipynb)
  2. [Sequence Classification: Sentiment](tutorials/2.%20Sequence%20Classification)
      - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/2.%20Sequence%20Classification/sequence_classification.ipynb)
  3. [Embeddings: Transformer Embeddings e.g. BERT, XLM, GPT2, XLNet, roBERTa, ALBERT](tutorials/3.%20Embeddings)
      - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/3.%20Embeddings/embeddings.ipynb)
  4. [Question Answering: Span-based Question Answering Model](tutorials/4.%20Question%20Answering)
      - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/4.%20Question%20Answering/question_answering.ipynb)
  5. [Custom Fine-Tuning and Training with Transformer Models](tutorials/5.%20Finetuning%20and%20Training%20(Advanced))
      - Training a Sequence Classifier
        - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/5.%20Finetuning%20and%20Training%20(Advanced)/sequence_classification_training.ipynb)
      - Fine-tuning a Transformers Language Model
        - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novetta/adaptnlp/blob/master/tutorials/5.%20Finetuning%20and%20Training%20(Advanced)/fine_tuning.ipynb)
  
Checkout the [documentation](https://novetta.github.io/adaptnlp) for more information.
  
## REST Service 

We use FastAPI for standing up endpoints for serving state-of-the-art NLP models with AdaptNLP.

![Swagger Example](https://raw.githubusercontent.com/novetta/adaptnlp/master/docs/img/fastapi-docs.png)

The [REST](https://github.com/Novetta/adaptnlp/tree/master/rest) directory contains more detail on deploying a REST API locally or with docker in a very easy and
fast way.
  
## Docker

AdaptNLP official docker images are up on [Docker Hub](https://hub.docker.com/r/achangnovetta/adaptnlp).

Images have AdaptNLP installed from source in developer mode with tutorial notebooks available.
 
Images can build with GPU support if NVIDA-Docker is correctly installed.

#### Pull and Run AdaptNLP Immediately
Simply run an image with AdaptNLP installed from source in developer mode by running:
```
docker run -it --rm achangnovetta/adaptnlp:latest
```
Run an image with AdaptNLP running on GPUs if you have nvidia drivers and nvidia-docker 19.03+ installed:
```
docker run -it --rm --gpus all achangnovetta/adaptnlp:latest
```

#### Build

Build docker image and run container with the following commands in the directory of the Dockerfile
to create a container with adaptnlp installed and ready to go

Note: A container with GPUs enabled requires Docker version 19.03+ and nvida-docker installed
```
docker build -t achangnovetta/adaptnlp:latest .
docker run -it --rm achangnovetta/adaptnlp:latest
```
If you want to use CUDA compatible GPUs 
```
docker run -it --rm --gpus all achangnovetta/adaptnlp:latest
```

## Contact

Please contact the author Andrew Chang at achang@novetta.com with questions or comments regarding AdaptNLP.

Follow  us on Twitter at [@achang1618](https://twitter.com/achang1618) and [@AdaptNLP](https://twitter.com/AdaptNLP) for
updates and NLP dialogue.

## License

This project is licensed under the terms of the Apache 2.0 license.
 



