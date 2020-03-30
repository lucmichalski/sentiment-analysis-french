This advanced tutorial section goes over using AdaptNLP for training and fine-tuning your own custom NLP models
to get State-of-the-Art results.

You should ideally follow the tutorials along with the provided notebooks in the `tutorials` directory at the top
level of the AdaptNLP library.

You could also run the code snippets in these tutorials straight through the python interpreter as well.

## Install and Setup

AdaptNLP can be used with or without GPUs.  AdaptNLP will automatically make use of GPU VRAM in environment with
CUDA-compatible NVIDIA GPUs and NVIDIA drivers installed.  GPU-less environments will run AdaptNLP modules fine as well.

You will almost always want to utilize GPUs for training and fine-tuning useful NLP models, so a CUDA-compatible NVIDIA
GPU is a must.

Multi-GPU environments with Apex installed can allow for distributed and/or mixed precision training.

## Overview of Training and Finetuning Capabilities

Simply training a state-of-the-art sequence classification model can be done with AdaptNLP using Flair's sequence 
classification model and trainer with general pre-trained language models.  With encoders providing accurate word 
representations via. models like ALBERT, GPT2, and other transformer models, we can produce accurate NLP task-related
models.

With the concepts of [ULMFiT](https://arxiv.org/abs/1801.06146) in mind, AdaptNLP's approach in training downstream
predictive NLP models like sequence classification takes a step further than just utilizing pre-trained
contextualized embeddings.  We are able to effectively fine-tune state-of-the-art language models for useful NLP
tasks on various domain specific data.

##### Training a Sequence Classification with `SequenceClassifierTrainer`

```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer
from flair.datasets import TREC_6

# Specify directory for trainer files and model to be downloaded to
OUTPUT_DIR = "Path/to/model/output/directory" 

# Load corpus and instantiate AdaptNLP's `EasyDocumentEmbeddings` with desired embeddings
corpus = TREC_6() # Or path to directory of train.csv, test.csv, dev.csv files at "Path/to/data/directory" 
doc_embeddings = EasyDocumentEmbeddings("bert-base-cased", methods=["rnn"])

# Instantiate the trainer for Sequence Classification with the dataset, embeddings, and mapping of column index of data
sc_trainer = SequenceClassifierTrainer(corpus=corpus, encoder=doc_embeddings)

# Find optimal learning rate with automated learning rate finder
learning_rate = sc_trainer.find_learning_rate(output_dir=OUTPUT_DIR)

# Train the sequence classifier
sc_trainer.train(output_dir=OUTPUT_DIR, learning_rate=learning_rate, mini_batch_size=32, max_epochs=150)

# Now load the `EasySequenceClassifier` with the path to your trained model and run inference on your text.
from adaptnlp import EasySequenceClassifier

example_text = '''Where was the Queen's wedding held? '''

classifier = EasySequenceClassifier()

sentences = classifier.tag_text(example_text, model_name_or_path=OUTPUT_DIR)
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

##### Fine-Tuning a Transformers Language Model with `LMFineTuner`


```python
from adaptnlp import LMFineTuner

# Set output directory to store fine-tuner files and models
OUTPUT_DIR = "Path/to/model/output/directory" 

# Set path to train.csv and test.csv datasets, must have a column header labeled "text" to specify data to train language model
train_data_file = "Path/to/train.csv" 
eval_data_file = "Path/to/test.csv"

finetuner = LMFineTuner(
                        train_data_file=train_data_file,
                        eval_data_file=eval_data_file,
                        model_type="bert",
                        model_name_or_path="bert-base-cased",
                        mlm=True
                        )
# Freeze layers up to the last group of classification layers
finetuner.freeze()

# Find optimal learning rate with automated learning rate finder
learning_rate = finetuner.find_learning_rate(base_path=OUTPUT_DIR)
finetuner.freeze()

finetuner.train_one_cycle(
                          output_dir=OUTPUT_DIR,
                          learning_rate=learning_rate,
                          per_gpu_train_batch_size=4,
                          num_train_epochs=10.0,
                          evaluate_during_training=True,
                          )
```