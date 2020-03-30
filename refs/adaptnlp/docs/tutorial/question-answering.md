Question Answering is the NLP task of producing a legible answer from being provided two text inputs: the context
and the question in regards to the context.

Examples of Question Answering models are span-based models that output a start and end index that outline the relevant
"answer" from the context provided.  With these models, we can extract answers from various questions and queries
regarding any unstructured text.

Below, we'll walk through how we can use AdaptNLP's `EasyQuestionAnswering` module to extract span-based text answers
from unstructured text using state-of-the-art question answering models.


## Getting Started with `EasyQuestionAnswering`

You can use `EasyQuestionAnswering` to run span-based question answering models.

Providing a context and query, we get an output of top `n_best_size` answer predictions along with token span indices 
and probability scores.

We'll first get started by importing the `EasyQuestionAnswering` class from AdaptNLP.

After that, we set some example text that we'll use further down and then instantiate the QA model.

```python
from adaptnlp import EasyQuestionAnswering

text = """Amazon.com, Inc.[6] (/ˈæməzɒn/), is an American multinational technology company based in Seattle, 
Washington that focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. 
It is considered one of the Big Four technology companies along with Google, Apple, and Facebook.[7][8][9] 
Amazon is known for its disruption of well-established industries through technological innovation and mass 
scale.[10][11][12] It is the world's largest e-commerce marketplace, AI assistant provider, and cloud computing 
platform[13] as measured by revenue and market capitalization.[14] Amazon is the largest Internet company by 
revenue in the world.[15] It is the second largest private employer in the United States[16] and one of the world's 
most valuable companies. Amazon is the second largest technology company by revenue. Amazon was founded by Jeff Bezos 
on July 5, 1994, in Bellevue, Washington. The company initially started as an online marketplace for books but later 
expanded to sell electronics, software, video games, apparel, furniture, food, toys, and jewelry. In 2015, Amazon 
surpassed Walmart as the most valuable retailer in the United States by market capitalization.[17] In 2017, Amazon 
acquired Whole Foods Market for $13.4 billion, which vastly increased Amazon's presence as a brick-and-mortar 
retailer.[18] In 2018, Bezos announced that its two-day delivery service, Amazon Prime, had surpassed 100 million 
subscribers worldwide
"""

qa_model = EasyQuestionAnswering()
```
### Running inference with `predict_qa(query: Union[List[str], str], context: Union[List[str], str], n_best_size: int, mini_batch_size: int, model_name_or_path: str, **kwargs)`

Now that we have the question answering model instantiated, we can run inference using the built-in `predict_qa` method.

!!! note
    You can set `model_name_or_path` to any of Transformer's pretrained question answering models.
    Transformers models are located at [https://huggingface.co/models](https://huggingface.co/models).  You can also pass in
    the path of a custom trained Transformers `xxxForQuestionAnswering` model.

Here is an example of running inference on Transformer's DistilBERT QA model fine-tuned on SQUAD:

```python
top_prediction, all_nbest_json = qa_model.predict_qa(query="What does Amazon do?", context=text, n_best_size=5, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")


print(top_prediction)

print(all_nbest_json)
```

We can do the same thing but now more question/context pairs!

```python
questions = ["What does Amazon do?",
             "What happened July 5, 1994?",
             "How much did Amazon acquire Whole Foods for?"]
 
top_prediction, all_nbest_json = qa_model.predict_qa(query=questions, context=[text]*3, n_best_size=5, mini_batch_size=1, model_name_or_path="distilbert-base-uncased-distilled-squad")


print(top_prediction)

print(all_nbest_json)
```

!!! note
    Check out `TransformersQuestionAnswering` for a more in-depth look into the additional parameters you can pass into
    the `EasyQuestionAnswering.predict_qa` method.
    XLNET and XLM models will be supported in the near future.
