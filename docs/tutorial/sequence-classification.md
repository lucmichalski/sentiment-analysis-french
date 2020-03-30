Sequence Classification (or Text Classification) is the NLP task of predicting a label for a sequence of words.

A sentiment classifier is an example of a sequence classification model.

Below, we'll walk through how we can use AdaptNLP's `EasySequenceClassification` module to label unstructured text with
state-of-the-art sequence classification models.


## Getting Started with `EasySequenceClassifier`

We'll first get started by importing the `EasySequenceClassifier` class from AdaptNLP.

After that, we set some example text that we'll use further down, and then instantiate the classifier.

```python
from adaptnlp import EasySequenceClassifier 

review_text = ["That wasn't very good.",
               "I really liked it",
               "It was really useful",
               "It broke after I bought it."]

movie_sentiment_text = '''Novetta Solutions is the best. Albert Einstein used to be employed at Novetta Solutions. 
The Wright brothers loved to visit the JBF headquarters, and they would have a chat with Albert.'''

classifier = EasySequenceClassifier()
```

### Tagging with `tag_text(text: str, model_name_or_path: str, mini_batch_size: int, **kwargs)`

Now that we have the classifier instantiated, we are ready to load in a sequence classification model and tag the text
with the built-in `tag_text()` method.  

This method takes in parameters: `text`, `model_name_or_path`, and `mini_batch_size` as well as optional keyword arguments.

!!! note 
    You can set `model_name_or_path` to any of Transformers or Flair's pretrained sequence classification models.
    Transformers models are located at [https://huggingface.co/models](https://huggingface.co/models).  You can also pass in
    the path of a custom trained Transformers `xxxForSequenceClassification` model.
 
The method returns a list of Flair's Sentence objects.

Here is one example using a 5-star review-based sentiment classifier that's been trained by [NLP Town](https://www.nlp.town/):

```python
# Predict
sentences = classifier.tag_text(review_text, model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment", mini_batch_size=1)

print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

Another example is shown below with a Flair's pre-trained sentiment classifier:

!!! note
    Additional keyword arguments can be passed in as parameters for Flair's token tagging `predict()` method i.e. 
    `mini_batch_size`, `embedding_storage_mode`, `verbose`, etc.

```python
# Predict
sentences = classifier.tag_text(movie_sentiment_text, model_name_or_path="en-sentiment")

print("Label output:\n")
for sentence in sentences:
    print(sentence.labels)
```

All of Flair's pretrained sequence classifiers are available for loading through the `model_name_or_path` parameter, 
and they can be found [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

A path to a custom trained Flair Sequence Classifier can also be passed through the `model_name_or_path` param.

Note: You can run `tag_all()` with the sequence classifier, as detailed in the Token Tagging Tutorial.
