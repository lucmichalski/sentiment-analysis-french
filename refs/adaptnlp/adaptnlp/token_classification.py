from typing import List, Dict, Union
from collections import defaultdict

from flair.data import Sentence
from flair.models import SequenceTagger


class EasyTokenTagger:
    """ Token level classification models

    Usage:

    ```python
    >>> tagger = adaptnlp.EasyTokenTagger()
    >>> tagger.tag_text(text="text you want to tag", model_name_or_path="ner-ontonotes")
    ```
    """

    def __init__(self):
        self.token_taggers: Dict[SequenceTagger] = defaultdict(bool)

    def tag_text(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        model_name_or_path: str = "ner-ontonotes",
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags tokens with labels the token classification models have been trained on

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **model_name_or_path** - The hosted model name key or model path
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - Keyword arguments for Flair's `SequenceTagger.predict()` method
        **return** - A list of Flair's `Sentence`'s
        """
        # Load Sequence Tagger Model and Pytorch Module into tagger dict
        if not self.token_taggers[model_name_or_path]:
            self.token_taggers[model_name_or_path] = SequenceTagger.load(
                model_name_or_path
            )

        tagger = self.token_taggers[model_name_or_path]
        return tagger.predict(
            sentences=text,
            mini_batch_size=mini_batch_size,
            use_tokenizer=True,
            **kwargs,
        )

    def tag_all(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags tokens with all labels from all token classification models

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - Keyword arguments for Flair's `SequenceTagger.predict()` method
        **return** A list of Flair's `Sentence`'s
        """
        if len(self.token_taggers) == 0:
            print("No token classification models loaded...")
            return Sentence()
        sentences = text
        for tagger_name in self.token_taggers.keys():
            sentences = self.tag_text(
                sentences,
                model_name_or_path=tagger_name,
                mini_batch_size=mini_batch_size,
                **kwargs,
            )
        return sentences
