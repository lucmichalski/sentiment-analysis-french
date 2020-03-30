from typing import Union, List
from pathlib import Path
from abc import ABC, abstractmethod

from flair.data import Sentence


class AdaptiveModel(ABC):
    @abstractmethod
    def load(
        self, model_name_or_path: Union[str, Path],
    ):
        """ Load model into the `AdaptiveModel` object as alternative constructor """
        pass

    @abstractmethod
    def predict(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Run inference on the model """
        pass
