from typing import List, Dict, Union
from collections import defaultdict

from flair.data import Sentence
from flair.embeddings import (
    Embeddings,
    # WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    BertEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    OpenAIGPT2Embeddings,
    XLNetEmbeddings,
    XLMEmbeddings,
    RoBERTaEmbeddings,
    # MuseCrosslingualEmbeddings,
)

FLAIR_PRETRAINED_MODEL_NAMES = {
    "multi-forward",
    "multi-backward",
    "multi-v0-forward",
    "multi-v0-backward",
    "multi-v0-forward-fast",
    "multi-v0-backward-fast",
    "en-forward",
    "en-backward",
    "en-forward-fast",
    "en-backward-fast",
    "news-forward",
    "news-backward",
    "news-forward-fast",
    "news-backward-fast",
    "mix-forward",
    "mix-backward",
    "ar-forward",
    "ar-backward",
    "bg-forward-fast",
    "bg-backward-fast",
    "bg-forward",
    "bg-backward",
    "cs-forward",
    "cs-backward",
    "cs-v0-forward",
    "cs-v0-backward",
    "da-forward",
    "da-backward",
    "de-forward",
    "de-backward",
    "de-historic-ha-forward",
    "de-historic-ha-backward",
    "de-historic-wz-forward",
    "de-historic-wz-backward",
    "es-forward",
    "es-backward",
    "es-forward-fast",
    "es-backward-fast",
    "eu-forward",
    "eu-backward",
    "eu-v1-forward",
    "eu-v1-backward",
    "eu-v0-forward",
    "eu-v0-backward",
    "fa-forward",
    "fa-backward",
    "fi-forward",
    "fi-backward",
    "fr-forward",
    "fr-backward",
    "he-forward",
    "he-backward",
    "hi-forward",
    "hi-backward",
    "hr-forward",
    "hr-backward",
    "id-forward",
    "id-backward",
    "it-forward",
    "it-backward",
    "ja-forward",
    "ja-backward",
    "nl-forward",
    "nl-backward",
    "nl-v0-forward",
    "nl-v0-backward",
    "no-forward",
    "no-backward",
    "pl-forward",
    "pl-backward",
    "pl-opus-forward",
    "pl-opus-backward",
    "pt-forward",
    "pt-backward",
    "pubmed-forward",
    "pubmed-backward",
    "sl-forward",
    "sl-backward",
    "sl-v0-forward",
    "sl-v0-backward",
    "sv-forward",
    "sv-backward",
    "sv-v0-forward",
    "sv-v0-backward",
    "ta-forward",
    "ta-backward",
}


class EasyWordEmbeddings:
    """ Word embeddings from the latest language models

    Usage:

    ```python
    >>> embeddings = adaptnlp.EasyWordEmbeddings()
    >>> embeddings.embed_text("text you want embeddings for", model_name_or_path="bert-base-cased")
    ```
    """

    def __init__(self):
        self.models: Dict[Embeddings] = defaultdict(bool)

    def embed_text(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        model_name_or_path: str = "bert-base-cased",
    ) -> List[Sentence]:
        """ Produces embeddings for text

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **model_name_or_path** - The hosted model name key or model path
        **return** - A list of Flair's `Sentence`s
        """
        # Convert into sentences
        if isinstance(text, str):
            sentences = Sentence(text)
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            sentences = [Sentence(t) for t in text]
        else:
            sentences = text

        # Load correct Embeddings module
        if not self.models[model_name_or_path]:
            if "bert" in model_name_or_path and "roberta" not in model_name_or_path:
                self.models[model_name_or_path] = BertEmbeddings(model_name_or_path)
            elif "roberta" in model_name_or_path:
                self.models[model_name_or_path] = RoBERTaEmbeddings(model_name_or_path)
            elif "gpt2" in model_name_or_path:
                self.models[model_name_or_path] = OpenAIGPT2Embeddings(
                    model_name_or_path
                )
            elif "xlnet" in model_name_or_path:
                self.models[model_name_or_path] = XLNetEmbeddings(model_name_or_path)
            elif "xlm" in model_name_or_path:
                self.models[model_name_or_path] = XLMEmbeddings(model_name_or_path)
            elif (
                "flair" in model_name_or_path
                or model_name_or_path in FLAIR_PRETRAINED_MODEL_NAMES
            ):
                self.models[model_name_or_path] = FlairEmbeddings(model_name_or_path)
            else:
                print(
                    f"Corresponding flair embedding module not found for {model_name_or_path}"
                )
                return Sentence("")
        embedding = self.models[model_name_or_path]
        return embedding.embed(sentences)

    def embed_all(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        *model_names_or_paths: str,
    ) -> List[Sentence]:
        """Embeds text with all embedding models loaded

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **model_names_or_paths** -  A variable input of model names or paths to embed
        **return** - A list of Flair's `Sentence`s
        """
        # Convert into sentences
        if isinstance(text, str):
            sentences = Sentence(text)
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            sentences = [Sentence(t) for t in text]
        else:
            sentences = text

        if model_names_or_paths:
            for embedding_name in model_names_or_paths:
                sentences = self.embed_text(
                    sentences, model_name_or_path=embedding_name
                )
        else:
            for embedding_name in self.models.keys():
                sentences = self.embed_text(
                    sentences, model_name_or_path=embedding_name
                )
        return sentences


class EasyStackedEmbeddings:
    """ Word Embeddings that have been concatenated and "stacked" as specified by flair

    Usage:

    ```python
    >>> embeddings = adaptnlp.EasyStackedEmbeddings("bert-base-cased", "gpt2", "xlnet-base-cased")
    ```

    **Parameters:**

    * **&ast;embeddings** - Non-keyword variable number of strings specifying the embeddings you want to stack
    """

    def __init__(self, *embeddings: str):
        print("May need a couple moments to instantiate...")
        self.embedding_stack = []

        # Load correct Embeddings module
        for model_name_or_path in embeddings:
            if "bert" in model_name_or_path and "roberta" not in model_name_or_path:
                self.embedding_stack.append(BertEmbeddings(model_name_or_path))
            elif "roberta" in model_name_or_path:
                self.embedding_stack.append(RoBERTaEmbeddings(model_name_or_path))
            elif "gpt2" in model_name_or_path:
                self.embedding_stack.append(OpenAIGPT2Embeddings(model_name_or_path))
            elif "xlnet" in model_name_or_path:
                self.embedding_stack.append(XLNetEmbeddings(model_name_or_path))
            elif "xlm" in model_name_or_path:
                self.embedding_stack.append(XLMEmbeddings(model_name_or_path))
            elif (
                "flair" in model_name_or_path
                or model_name_or_path in FLAIR_PRETRAINED_MODEL_NAMES
            ):
                self.embedding_stack.append(FlairEmbeddings(model_name_or_path))
            else:
                print(
                    f"Corresponding flair embedding module not found for {model_name_or_path}"
                )

        assert len(self.embedding_stack) != 0
        self.stacked_embeddings = StackedEmbeddings(embeddings=self.embedding_stack)

    def embed_text(
        self, text: Union[List[Sentence], Sentence, List[str], str],
    ) -> List[Sentence]:
        """ Stacked embeddings

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        **return** A list of Flair's `Sentence`s
        """
        # Convert into sentences
        if isinstance(text, str):
            sentences = [Sentence(text)]
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            sentences = [Sentence(t) for t in text]
        elif isinstance(text, Sentence):
            sentences = [text]
        else:
            sentences = text

        # Unlike flair embeddings modules, stacked embeddings do not return a list of sentences
        self.stacked_embeddings.embed(sentences)
        return sentences


class EasyDocumentEmbeddings:
    """ Document Embeddings generated by pool and rnn methods applied to the word embeddings of text

    Usage:

    ```python
    >>> embeddings = adaptnlp.EasyDocumentEmbeddings("bert-base-cased", "xlnet-base-cased", methods["rnn"])
    ```

    **Parameters:**

    * **&ast;embeddings** - Non-keyword variable number of strings referring to model names or paths
    * **methods** - A list of strings to specify which document embeddings to use i.e. ["rnn", "pool"] (avoids unncessary loading of models if only using one)
    * **configs** - A dictionary of configurations for flair's rnn and pool document embeddings
    ```python
    >>>example_configs = {"pool_configs": {"fine_tune_mode": "linear", "pooling": "mean", },
    ...                   "rnn_configs": {"hidden_size": 512,
    ...                                   "rnn_layers": 1,
    ...                                   "reproject_words": True,
    ...                                   "reproject_words_dimension": 256,
    ...                                   "bidirectional": False,
    ...                                   "dropout": 0.5,
    ...                                   "word_dropout": 0.0,
    ...                                   "locked_dropout": 0.0,
    ...                                   "rnn_type": "GRU",
    ...                                   "fine_tune": True, },
    ...                  }
    ```
    """

    __allowed_methods = ["rnn", "pool"]
    __allowed_configs = ("pool_configs", "rnn_configs")

    def __init__(
        self,
        *embeddings: str,
        methods: List[str] = ["rnn", "pool"],
        configs: Dict = {
            "pool_configs": {"fine_tune_mode": "linear", "pooling": "mean"},
            "rnn_configs": {
                "hidden_size": 512,
                "rnn_layers": 1,
                "reproject_words": True,
                "reproject_words_dimension": 256,
                "bidirectional": False,
                "dropout": 0.5,
                "word_dropout": 0.0,
                "locked_dropout": 0.0,
                "rnn_type": "GRU",
                "fine_tune": True,
            },
        },
    ):
        print("May need a couple moments to instantiate...")
        self.embedding_stack = []

        # Check methods
        for m in methods:
            assert m in self.__class__.__allowed_methods

        # Set configs for pooling and rnn parameters
        for k, v in configs.items():
            assert k in self.__class__.__allowed_configs
            setattr(self, k, v)

        # Load correct Embeddings module
        for model_name_or_path in embeddings:
            if "bert" in model_name_or_path and "roberta" not in model_name_or_path:
                self.embedding_stack.append(BertEmbeddings(model_name_or_path))
            elif "roberta" in model_name_or_path:
                self.embedding_stack.append(RoBERTaEmbeddings(model_name_or_path))
            elif "gpt2" in model_name_or_path:
                self.embedding_stack.append(OpenAIGPT2Embeddings(model_name_or_path))
            elif "xlnet" in model_name_or_path:
                self.embedding_stack.append(XLNetEmbeddings(model_name_or_path))
            elif "xlm" in model_name_or_path:
                self.embedding_stack.append(XLMEmbeddings(model_name_or_path))
            elif (
                "flair" in model_name_or_path
                or model_name_or_path in FLAIR_PRETRAINED_MODEL_NAMES
            ):
                self.embedding_stack.append(FlairEmbeddings(model_name_or_path))
            else:
                print(
                    f"Corresponding flair embedding module not found for {model_name_or_path}"
                )

        assert len(self.embedding_stack) != 0
        if "pool" in methods:
            self.pool_embeddings = DocumentPoolEmbeddings(
                self.embedding_stack, **self.pool_configs
            )
            print("Pooled embedding loaded")
        if "rnn" in methods:
            self.rnn_embeddings = DocumentRNNEmbeddings(
                self.embedding_stack, **self.rnn_configs
            )
            print("RNN embeddings loaded")

    def embed_pool(
        self, text: Union[List[Sentence], Sentence, List[str], str],
    ) -> List[Sentence]:
        """ Stacked embeddings


        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        **return** - A list of Flair's `Sentence`s
        """
        if isinstance(text, str):
            sentences = [Sentence(text)]
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            sentences = [Sentence(t) for t in text]
        elif isinstance(text, Sentence):
            sentences = [text]
        else:
            sentences = text
        self.pool_embeddings.embed(sentences)
        return sentences

    def embed_rnn(
        self, text: Union[List[Sentence], Sentence, List[str], str],
    ) -> List[Sentence]:
        """ Stacked embeddings

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        **return** - A list of Flair's `Sentence`s
        """
        if isinstance(text, str):
            sentences = [Sentence(text)]
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            sentences = [Sentence(t) for t in text]
        elif isinstance(text, Sentence):
            sentences = [text]
        else:
            sentences = text
        self.rnn_embeddings.embed(sentences)
        return sentences
