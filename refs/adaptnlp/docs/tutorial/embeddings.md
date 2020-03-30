Embeddings for NLP are the vector representations of unstructured text.

Examples of applications of Word Embeddings are downstream NLP task model training and similarity search. 

Below, we'll walk through how we can use AdaptNLP's `EasyWordEmbeddings`, `EasyStackedEmbeddings`, and 
`EasyDocumentEmbeddings` classes.

## Available Language Models

Huggingface's Transformer's model key shortcut names can be found [here](https://huggingface.co/transformers/pretrained_models.html).

The key shortcut names for their public model-sharing repository are available [here](https://huggingface.co/models) as of
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

## Getting Started with `EasyWordEmbeddings`

With `EasyWordEmbeddings`, you can load in a language model and produce contextual embeddings with text input.

You can look at each word's embeddings which have been contextualized by their surrounding text, meaning embedding
outputs will change for the same word depending on the text as a whole.

Below is an example of producing word embeddings from OpenAI's GPT2 language model.


```python
from adaptnlp import EasyWordEmbeddings

example_text = "This is Albert.  My last name is Einstein.  I like physics and atoms."

# Instantiate embeddings tagger
embeddings = EasyWordEmbeddings()

# Get GPT2 embeddings of example text... A list of flair Sentence objects are generated
sentences = embeddings.embed_text(example_text, model_name_or_path="gpt2")
# Iterate through to access the embeddings
for token in sentences[0]:
    print(token.get_embedding())
    break
```

## Getting Started with `EasyStackedEmbeddings`

Stacked embeddings are a simple yet important concept pointed out by Flair that can help produce state-of-the-art
results in downstream NLP models.

It produces contextualized word embeddings like `EasyWordEmbeddings`, except the embeddings are the concatenation of
tensors from multiple language model. 

Below is an example of producing stacked word embeddings from the BERT base cased and XLNet base cased language
models.  `EasyStackedEmbeddings` take in a variable number of key shortcut names to pre-trained language models.

```python
from adaptnlp import EasyStackedEmbeddings

# Instantiate stacked embeddings tagger
embeddings = EasyStackedEmbeddings("bert-base-cased", "xlnet-base-cased")

# Run the `embed_stack` method to get the stacked embeddings outlined above
sentences = embeddings.embed_text(example_text)
# Iterate through to access the embeddings
for token in sentences[0]:
    print(token.get_embedding())
    break
```

## Getting Started with `EasyDocumentEmbeddings`

Document embeddings you can load in a variable number of language models, just like in stacked embeddings, and produce
and embedding for an entire text.  Unlike `EasyWordEmbeddings` and `EasyStackedEmbeddings`, `EasyDocumentEmbeddings`
will produce one contextualized embedding for a sequence of words using the pool or RNN method provided by Flair.

If you are familiar with using Flair's RNN document embeddings, you can pass in hyperparameters through the `config`
parameter when instantiating an `EasyDocumentEmbeddings` object.

Below is an example of producing aan embedding from the entire text using the BERT base cased and XLNet base 
cased language models.  We also show the embeddings you get using the pool or RNN method.


```python
from adaptnlp import EasyDocumentEmbeddings

# Instantiate document embedder with stacked embeddings
embeddings = EasyDocumentEmbeddings("bert-base-cased", "xlnet-base-cased")

# Document Pool embedding...Instead of a list of flair Sentence objects, we get one Sentence object: the document
text = embeddings.embed_pool(example_text)
#get the text/document embedding
text[0].get_embedding()

# Now again but with Document RNN embedding
text = embeddings.embed_rnn(example_text)
#get the text/document embedding
text[0].get_embedding()
```