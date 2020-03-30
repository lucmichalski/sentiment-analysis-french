import logging
from typing import List, Dict, Union, Tuple
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader
from flair.data import Sentence, DataPoint, Label
from flair.models import TextClassifier
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
    BertForSequenceClassification,
    XLNetForSequenceClassification,
    AlbertForSequenceClassification,
)

from tqdm import tqdm

from adaptnlp.model import AdaptiveModel

logger = logging.getLogger(__name__)


class TransformersSequenceClassifier(AdaptiveModel):
    """ Adaptive model for Transformer's Sequence Classification Model

    Usage:
    ```python
    >>> classifier = TransformersSequenceClassifier.load("transformers-sc-model")
    >>> classifier.predict(text="Example text", mini_batch_size=32)
    ```

    **Parameters:**

    * **tokenizer** - A tokenizer object from Huggingface's transformers (TODO)and tokenizers
    * **model** - A transformers Sequence Classsifciation model
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        # Load up model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

        # Setup cuda and automatic allocation of model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def load(cls, model_name_or_path: str) -> AdaptiveModel:
        """ Class method for loading and constructing this classifier

        * **model_name_or_path** - A key string of one of Transformer's pre-trained Sequence Classifier Model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        classifier = cls(tokenizer, model)
        return classifier

    def predict(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        use_tokenizer: bool = True,
        **kwargs,
    ) -> List[Sentence]:
        """ Predict method for running inference using the pre-trained sequence classifier model

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers classifier
        """
        id2label = self.model.config.id2label
        sentences = text
        results: List[Sentence] = []

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, DataPoint) or isinstance(sentences, str):
                sentences = [sentences]

            # filter empty sentences
            if isinstance(sentences[0], Sentence):
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
            if len(sentences) == 0:
                return sentences

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[DataPoint, str]] = [
                sentences[index] for index in rev_order_len_index
            ]
            # Turn all Sentence objects into strings
            if isinstance(reordered_sentences[0], Sentence):
                str_reordered_sentences = [
                    sentence.to_original_text() for sentence in sentences
                ]
            else:
                str_reordered_sentences = reordered_sentences

            # Tokenize and get dataset
            dataset = self._tokenize(str_reordered_sentences)
            dataloader = DataLoader(dataset, batch_size=mini_batch_size)
            predictions: List[Tuple[str, float]] = []

            logger.info(f"Running prediction on {len(dataset)} text sequences")
            logger.info(f"Batch size = {mini_batch_size}")
            for batch in tqdm(dataloader, desc="Predicting text"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                if len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                outputs = self.model(**inputs)
                logits = outputs[0]
                preds = torch.softmax(logits, dim=1).tolist()

                predictions += preds

            for text, pred in zip(str_reordered_sentences, predictions):
                # Initialize and assign labels to each class in each datapoint prediction
                text_sent = Sentence(text)
                for k, v in id2label.items():
                    label = Label(value=v, score=pred[k])
                    text_sent.add_label(label)
                results.append(text_sent)

        # Order results back into original order
        results = [results[index] for index in original_order_index]

        return results

    def _tokenize(
        self, sentences: Union[List[Sentence], Sentence, List[str], str]
    ) -> TensorDataset:
        """ Batch tokenizes text and produces a `TensorDatset` with them """

        tokenized_text = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if isinstance(
            self.model,
            (
                BertForSequenceClassification,
                XLNetForSequenceClassification,
                AlbertForSequenceClassification,
            ),
        ):
            dataset = TensorDataset(
                tokenized_text["input_ids"],
                tokenized_text["attention_mask"],
                tokenized_text["token_type_ids"],
            )
        else:
            dataset = TensorDataset(
                tokenized_text["input_ids"], tokenized_text["attention_mask"]
            )

        return dataset


class FlairSequenceClassifier(AdaptiveModel):
    """ Adaptive Model for Flair's Sequence Classifier...very basic

    Usage:
    ```python
    >>> classifier = FlairSequenceClassifier.load("en-sentiment")
    >>> classifier.predict(text="Example text", mini_batch_size=32)
    ```

    **Parameters:**

    * **model_name_or_path** - A key string of one of Flair's pre-trained Sequence Classifier Model
    """

    def __init__(self, model_name_or_path: str):
        self.classifier = TextClassifier.load(model_name_or_path)

    @classmethod
    def load(cls, model_name_or_path: str) -> AdaptiveModel:
        """ Class method for loading a constructing this classifier

        * **model_name_or_path** - A key string of one of Flair's pre-trained Sequence Classifier Model
        """
        classifier = cls(model_name_or_path)
        return classifier

    def predict(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        use_tokenizer=True,
        **kwargs,
    ) -> List[Sentence]:
        """ Predict method for running inference using the pre-trained sequence classifier model

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Flair classifier
        """
        return self.classifier.predict(
            sentences=text,
            mini_batch_size=mini_batch_size,
            use_tokenizer=use_tokenizer,
            **kwargs,
        )


class EasySequenceClassifier:
    """ Sequence classification models

    Usage:

    ```python
    >>> classifier = EasySequenceClassifier()
    >>> classifier.tag_text(text="text you want to label", model_name_or_path="en-sentiment")
    ```

    """

    def __init__(self):
        self.sequence_classifiers: Dict[AdaptiveModel] = defaultdict(bool)

    def tag_text(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        model_name_or_path: str = "en-sentiment",
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags a text sequence with labels the sequence classification models have been trained on

        * **text** - String, list of strings, `Sentence`, or list of `Sentence`s to be classified
        * **model_name_or_path** - The model name key or model path
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - (Optional) Keyword Arguments for Flair's `TextClassifier.predict()` method params
        **return** A list of Flair's `Sentence`'s
        """
        # Load Text Classifier Model and Pytorch Module into tagger dict
        if not self.sequence_classifiers[model_name_or_path]:
            """
            self.sequence_classifiers[model_name_or_path] = TextClassifier.load(
                model_name_or_path
            )
            """
            # TODO: Find an alternative model-check method like an `is_available(model_name_or_path)
            # Check whether this is a Transformers or Flair Sequence Classifier model we're loading
            try:
                self.sequence_classifiers[
                    model_name_or_path
                ] = FlairSequenceClassifier.load(model_name_or_path)
            except FileNotFoundError:
                logger.info(
                    f"{model_name_or_path} not a valid Flair pre-trained model...checking transformers repo"
                )
                try:
                    self.sequence_classifiers[
                        model_name_or_path
                    ] = TransformersSequenceClassifier.load(model_name_or_path)
                except ValueError:
                    logger.info("Try transformers")
                    return [Sentence("")]

        classifier = self.sequence_classifiers[model_name_or_path]
        return classifier.predict(text=text, mini_batch_size=mini_batch_size, **kwargs,)

    def tag_all(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags text with all labels from all sequence classification models

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - (Optional) Keyword Arguments for Flair's `TextClassifier.predict()` method params
        * **return** - A list of Flair's `Sentence`'s
        """
        sentences = text
        for tagger_name in self.sequence_classifiers.keys():
            sentences = self.tag_text(
                sentences,
                model_name_or_path=tagger_name,
                mini_batch_size=mini_batch_size,
                **kwargs,
            )
        return sentences
