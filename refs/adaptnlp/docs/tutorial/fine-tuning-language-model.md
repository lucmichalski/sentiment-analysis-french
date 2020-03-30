Fine-tuning a language model comes in handy when data of a target task comes from a different distribution compared
to the general-domain data that was used for pretraining a language model.

When fine-tuning the language model on data from a target task, the general-domain pretrained model is able to converge
quickly and adapt to the idiosyncrasies of the target data.  This can be seen from the efforts of ULMFiT and Jeremy
Howard's and Sebastian Ruder's approach on NLP transfer learning.

With AdaptNLP's `LMFineTuner`, we can start to fine-tune state-of-the-art pretrained transformers architecture 
language models provided by Huggingface's Transformers library.

Below are the available transformers language models for fine-tuning with `LMFineTuner`

| Transformer Model| Model Type/Architecture String Key|
| ------------- | ----------------------  |
| ALBERT | "albert" |
| DistilBERT | "distilbert" |
| BERT | "bert" |
| CamemBERT | "camembert" |
| RoBERTa | "roberta" |
| GPT | "gpt" |
| GPT2 | "gpt2" |

You can fine-tune on any transformers language models with the above architecture in Huggingface's Transformers
library.  Key shortcut names are located [here](https://huggingface.co/transformers/pretrained_models.html).

The same goes for Huggingface's public model-sharing repository, which is available [here](https://huggingface.co/models)
as of v2.2.2 of the Transformers library.

## Getting Started with `LMFineTuner`

You first want to specify the paths to train.csv and test.csv files that have header column labeled `text`.
You also want to specify the output directories for your fine-tuner and model.

```python

from adaptnlp import LMFineTuner

OUTPUT_DIR = "Path/to/model/output/directory"

train_data_file = "Path/to/train.csv" 
eval_data_file = "Path/to/test.csv"

```

We can then instantiate the language model fine-tuner by specifying which model type/architecture you want, and which
corresponding pretrained language model you would like to fine-tune on.

Here you can also specify whether the model relies on a causal or masked language modeling loss.

This is also where you would specify using distributed training and/or multi-precision.

We will freeze the model up to the language modeling classification group layer before fine-tuning.

```python

ft_configs = {
              "train_data_file": train_data_file,
              "eval_data_file": eval_data_file,
              "model_type": "bert",
              "model_name_or_path": "bert-base-cased",
              "mlm": True,
              "mlm_probability": 0.15,
              "config_name": None,
              "tokenizer_name": None,
              "cache_dir": None,
              "block_size": -1,
              "no_cuda": False,
              "overwrite_cache": False,
              "seed": 42,
              "fp16": False,
              "fp16_opt_level": "O1",
              "local_rank": -1,
             }
finetuner = LMFineTuner(**ft_configs)
finetuner.freeze()


```

We can then find the optimal learning rate with the help of the [cyclical learning rates method](https://arxiv.org/abs/1506.01186)
by Leslie Smith.

Using this along with our novel approach in [automatically extracting](https://forums.fast.ai/t/automated-learning-rate-suggester/44199?u=aychang)
an optimal learning rate, we can streamline training without pausing to manually extract the optimal learning rate.

The built-in `find_learning_rate()` will automatically reinitialize the parameteres and optimizer after running the
cyclical learning rates method.


```python
learning_rate_finder_configs = {
    "base_path": OUTPUT_DIR,
    "file_name": "learning_rate.tsv",
    "start_learning_rate": 1e-7,
    "end_learning_rate": 10,
    "iterations": 100,
    "mini_batch_size": 8,
    "stop_early": True,
    "smoothing_factor": 0.7,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
}
learning_rate = finetuner.find_learning_rate(**learning_rate_finder_configs)
finetuner.freeze()


```

We can now train and fine-tune the model like below.

```python
train_configs = {
    "output_dir": OUTPUT_DIR,
    "should_continue": False,
    "overwrite_output_dir": True,
    "evaluate_during_training": True,
    "per_gpu_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "learning_rate": learning_rate,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "num_train_epochs": 10.0,
    "max_steps": -1,
    "warmup_steps": 0,
    "logging_steps": 50,
    "save_steps": 50,
    "save_total_limit": None,
    "use_tensorboard": False,
}
finetuner.train_one_cycle(**train_configs)

```

Once the language model has been fine-tuned, we can load this model into an `EasyDocumentEmbeddings` object by 
specifying the path to the fine-tuned LM.  In this case, it is saved in `OUTPUT_DIR`.

This is what we do below.  Once we have the document embeddings instantiated, we can train a downstream custom sequence
classifier with embeddings produced from our fine-tuned language models.


```python
from adaptnlp import EasyDocumentEmbeddings

doc_embeddings = EasyDocumentEmbeddings(OUTPUT_DIR, methods = ["rnn"]) # We can specify to load the pool or rnn
                                                                             

```