A sequence classifier predicts a categorical label from the unstructured sequence of text that is provided as input.

AdaptNLP's `SequenceClassifierTrainer` uses Flair's sequence classification prediction head with Transformer and/or
Flair's contextualized embeddings.

You can specify the encoder you want to use from any of the following pretrained transformer language models provided
by Huggingface's Transformers library.  The model key shortcut names are located [here](https://huggingface.co/transformers/pretrained_models.html).

The key shortcut names of their public model-sharing repository are available [here](https://huggingface.co/models) as of
v2.2.2 of the Transformers library.


Below are the available transformers model architectures for use as an encoder:

| Transformer Model|
| -------------    |
| ALBERT |
| DistilBERT |
| BERT |
| CamemBERT |
| RoBERTa |
| GPT |
| GPT2 |
| XLNet |
| TransformerXL |
| XLM |
| XLMRoBERTa |

You can also use Flair's `FlairEmbeddings` who's model key shortcut names are located [here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)

You can also use AllenNLP's `ELMOEmbeddings` who's model key shortcut names are located [here](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/ELMO_EMBEDDINGS.md)

## Getting Started with `SequenceClassifierTrainer`

We want to start by specifying three things:
  1. `corpus`: A Flair `Corpus` data model object that contains train, test, and dev datasets.
    This can also be a path to a directory that contains `train.csv`, `test.csv`, and `dev.csv` files.
    If a path to the files is provided, you will require a `column_name_map` parameter that maps the indices of
    the `text` and `label` column headers i.e. {0: "text", 1: "label"} the colummn with text being at index 0 of the csv
  2. `output_dir`: A path to a directory to store trainer and model files
  3. `doc_embeddings`: The `EasyDocumentEmbeddings` object that has the specified key shortcut names to pretrained
    language models that the trainer will use as its encoder.

```python
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer
from flair.datasets import TREC_6

corpus = TREC_6() # Or path to directory of train.csv, test.csv, dev.csv files at "Path/to/data/directory" 

OUTPUT_DIR = "Path/to/model/output/directory" 

doc_embeddings = EasyDocumentEmbeddings("bert-base-cased", methods = ["rnn"]) # We can specify to load the pool or rnn
                                                                              # methods to avoid loading both.
```

Then we want to instantiate the trainer with the following parameters

```python
sc_configs = {
              "corpus": corpus,
              "encoder": doc_embeddings,
              "corpus_in_memory": True,
             }
sc_trainer = SequenceClassifierTrainer(**sc_configs)

```

We can then find the optimal learning rate with the help of the [cyclical learning rates method](https://arxiv.org/abs/1506.01186)
by Leslie Smith.

Using this along with our novel approach in [automatically extracting](https://forums.fast.ai/t/automated-learning-rate-suggester/44199?u=aychang)
an optimal learning rate, we can streamline training without pausing to manually extract the optimal learning rate.

The built-in `find_learning_rate()` will automatically reinitialize the parameteres and optimizer after running the
cyclical learning rates method.

```python
sc_lr_configs = {
        "output_dir": OUTPUT_DIR,
        "start_learning_rate": 1e-8,
        "end_learning_rate": 10,
        "iterations": 100,
        "mini_batch_size": 32,
        "stop_early": True,
        "smoothing_factor": 0.8,
        "plot_learning_rate": True,
}
learning_rate = sc_trainer.find_learning_rate(**sc_lr_configs)
```

We can then kick off training below.

```python

sc_train_configs = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": learning_rate,
        "mini_batch_size": 32,
        "anneal_factor": 0.5,
        "patience": 5,
        "max_epochs": 150,
        "plot_weights": False,
        "batch_growth_annealing": False,
}
sc_trainer.train(**sc_train_configs)

```

The model was saved in the directory `OUTPUT_DIR`.  We can load the sequence classifier into our `EasySequenceClassifier`
instance and start running inference.

```python
from adaptnlp import EasySequenceClassifier
# Set example text and instantiate tagger instance
example_text = '''Where was the Queen's wedding held? '''

classifier = EasySequenceClassifier()

sentences = classifier.tag_text(example_text, model_name_or_path=OUTPUT_DIR)
print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)

```

