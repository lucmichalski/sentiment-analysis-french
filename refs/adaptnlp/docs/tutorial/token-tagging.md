Token tagging (or token classification) is the NLP task of assigning a label to each individual word in the
provided text.

Examples of token tagging models are Named Entity Recognition(NER) and Parts of Speech(POS) models.  With these models,
we can generate tagged entities or parts of speech from unstructured text like "Persons" and "Nouns"

Below, we'll walk through how we can use AdaptNLP's `EasytokenTagger` module to label unstructured text with
state-of-the-art token tagging models.


## Getting Started with `EasyTokenTagger`

We'll first get started by importing the `EasyTokenTagger` module from AdaptNLP.

After that, we set some example text and instantiate the tagger.

```python
from adaptnlp import EasyTokenTagger

example_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

tagger = EasyTokenTagger()
```

### Tagging with `tag_text(text: str, model_name_or_path: str, **kwargs)`

Now that we have the tagger instantiated, we are ready to load in a token tagging model and tag the text with 
the built-in `tag_text()` method.  

This method takes in parameters: `text` and `model_name_or_path`.
 
The method returns a list of Flair's Sentence objects.

Note: Additional keyword arguments can be passed in as parameters for Flair's token tagging `predict()` method i.e. 
`mini_batch_size`, `embedding_storage_mode`, `verbose`, etc.

```python
# Tag the string
sentences = tagger.tag_text(text=example_text, model_name_or_path="ner-ontonotes")
```

All of Flair's pretrained token taggers are available for loading through the `model_name_or_path` parameter, 
and they can be found [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

A path to a custom trained Flair Token Tagger can also be passed through the `model_name_or_path` param.

Flair's pretrained token taggers (taken from the link above):

##### English Models
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner' | 4-class Named Entity Recognition |  Conll-03  |  **93.03** (F1) |
| 'ner-ontonotes' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.06** (F1) |
| 'chunk' |  Syntactic Chunking   |  Conll-2000     |  **96.47** (F1) |
| 'pos' |  Part-of-Speech Tagging |  Ontonotes     |  **98.6** (Accuracy) |
| 'frame'  |   Semantic Frame Detection |  Propbank 3.0     |  **97.54** (F1) |

##### Faster Models for CPU use
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-fast' | 4-class Named Entity Recognition |  Conll-03  |  **92.75** (F1) |
| 'ner-ontonotes-fast' | [18-class](https://spacy.io/api/annotation#named-entities) Named Entity Recognition |  Ontonotes  |  **89.27** (F1) |
| 'chunk-fast' |  Syntactic Chunking   |  Conll-2000     |  **96.22** (F1) |
| 'pos-fast' |  Part-of-Speech Tagging |  Ontonotes     |  **98.47** (Accuracy) |
| 'frame-fast'  |   Semantic Frame Detection | Propbank 3.0     |  **97.31** (F1) |

##### Multilingual Models
| ID | Task | Training Dataset | Accuracy |
| -------------    | ------------- |------------- |------------- |
| 'ner-multi' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **89.27**  (average F1) |
| 'ner-multi-fast' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **87.91**  (average F1) |
| 'ner-multi-fast-learn' | 4-class Named Entity Recognition |  Conll-03 (4 languages)  |  **88.18**  (average F1) |
| 'pos-multi' |  Part-of-Speech Tagging   |  Universal Dependency Treebank (12 languages)  |  **96.41** (average acc.) |
| 'pos-multi-fast' |  Part-of-Speech Tagging |  Universal Dependency Treebank (12 languages)  |  **92.88** (average acc.) |

##### German Models
| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'de-ner' | 4-class Named Entity Recognition |  Conll-03  |  **87.94** (F1) | |
| 'de-ner-germeval' | 4+4-class Named Entity Recognition |  Germeval  |  **84.90** (F1) | |
| 'de-pos' | Part-of-Speech Tagging |  UD German - HDT  |  **98.33** (Accuracy) | |
| 'de-pos-fine-grained' | Part-of-Speech Tagging |  German Tweets  |  **93.06** (Accuracy) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/pos-twitter-german) |

##### Other Languages
| ID | Task | Training Dataset | Accuracy | Contributor |
| -------------    | ------------- |------------- |------------- |------------- |
| 'fr-ner' | Named Entity Recognition |  [WikiNER (aij-wikiner-fr-wp3)](https://github.com/dice-group/FOX/tree/master/input/Wikiner)  |  **95.57** (F1) | [mhham](https://github.com/mhham) |
| 'nl-ner' | Named Entity Recognition |  [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)  |  **89.56** (F1) | [stefan-it](https://github.com/stefan-it/flair-experiments/tree/master/conll2002-ner-dutch) |
| 'da-ner' | Named Entity Recognition |  [Danish NER dataset](https://github.com/alexandrainst/danlp)  |   | [AmaliePauli](https://github.com/AmaliePauli) |
| 'da-pos' | Named Entity Recognition |  [Danish Dependency Treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md)  |  | [AmaliePauli](https://github.com/AmaliePauli) |

Now that the text has been tagged, take the returned sentences and see your results:

```python
# See Results
print("List string outputs of tags:\n")
for sen in sentences:
    print(sen.to_tagged_string())
```

If you want just the entities, you can run the below but you'll need to specify the `tag_type` as "ner" or "pos" etc.
(more information can be found in Flair's documentation):

```python
print("List entities tagged:\n")
for sen in sentences:
    for entity in sen.get_spans(tag_type="ner"):
        print(entity)
```

Here are some additional tag_types that support some of Flair's pre-trained token taggers:

| tag_type | Description |
| -------------    | ------------- |
| 'ner' | For Named Entity Recognition tagged text |
| 'pos' | For Parts of Speech tagged text |
| 'np' | For Syntactic Chunking tagged text |

NOTE: You can add your own tag_types when running the sequence classifier trainer in AdaptNLP.

### Tagging with `tag_all(text: str, model_name_or_path: str, **kwargs)`

As you tag text with multiple pretrained token tagging models, your tagger will have multiple models loaded...memory
permitting.

You can then use the built-in `tag_all()` method to tag your text with all models that are currently loaded in your
tagger.  See an example below:


```python
from adaptnlp import EasyTokenTagger

example_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

# Load models by tagging text
tagger = EasyTokenTagger()
tagger.tag_text(text=example_text, model_name_or_path="ner-ontonotes")
tagger.tag_text(text=example_text, model_name_or_path="pos")

# Now that the "pos" and "ner-ontonotes" models are loaded, run tag_all()
sentences = tagger.tag_all(text=example_text)
```

Now we can see below that you get a list of Flair sentences tagged with the "ner-ontonotes" AND "pos" model:

```python
print("List entities tagged:\n")print("List entities tagged:\n")
for sen in sentences:
    for entity in sen.get_spans(tag_type="pos"):
        print(entity)
        
for sen in sentences:
    for entity in sen.get_spans(tag_type="ner"):
        print(entity)
```


