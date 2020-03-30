from typing import Union, Dict
from pathlib import Path
import json
import csv

import numpy as np

from adaptnlp import EasyDocumentEmbeddings

from flair.datasets import CSVClassificationCorpus
from flair.data import Corpus
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter


class SequenceClassifierTrainer:
    """Sequence Classifier Trainer

    Usage:

    ```python
    >>> sc_trainer = SequenceClassifierTrainer(corpus="/Path/to/data/dir")
    ```

    **Parameters:**

    * **corpus** - A flair corpus data model or `Path`/string to a directory with train.csv/test.csv/dev.csv
    * **encoder** - A `EasyDocumentEmbeddings` object if training with a flair prediction head or `Path`/string if training with Transformer's prediction models
    * **column_name_map** - Required if corpus is not a `Corpus` object, it's a dictionary specifying the indices of the text and label columns of the csv i.e. {1:"text",2:"label"}
    * **corpus_in_memory** - Boolean for whether to store corpus embeddings in memory
    * **predictive_head** - For now either "flair" or "transformers" for the prediction head
    * **&ast;&ast;kwargs** - Keyword arguments for Flair's `TextClassifier` model class
    """

    def __init__(
        self,
        corpus: Union[Corpus, Path, str],
        encoder: Union[EasyDocumentEmbeddings, Path, str],
        column_name_map: None,
        corpus_in_memory: bool = True,
        predictive_head: str = "flair",
        **kwargs,
    ):
        if isinstance(corpus, Corpus):
            self.corpus = corpus
        else:
            if isinstance(corpus, str):
                corpus = Path(corpus)
            if not column_name_map:
                raise ValueError(
                    "If not instantiating with `Corpus` object, must pass in `column_name_map` argument to specify text/label indices"
                )
            self.corpus = CSVClassificationCorpus(
                corpus,
                column_name_map,
                skip_header=True,
                delimiter=",",
                in_memory=corpus_in_memory,
            )

        # Verify predictive head is within available heads
        self.available_predictive_head = ["flair", "transformers"]
        if predictive_head not in self.available_predictive_head:
            raise ValueError(
                f"predictive_head param must be one of the following: {self.available_predictive_head}"
            )
        self.predictive_head = predictive_head

        # Verify correct corresponding encoder is used with predictive head (This can be structured with better design in the future)
        if isinstance(encoder, EasyDocumentEmbeddings):
            if predictive_head == "transformers":
                raise ValueError(
                    "If using `transformers` predictive head, pass in the path to the transformer's model"
                )
            else:
                self.encoder = encoder
        else:
            if isinstance(encoder, str):
                encoder = Path(encoder)
            self.encoder = encoder

        # Create the label dictionary on init (store to keep from constantly generating label_dict) should we use dev/test set instead assuming all labels are provided?
        self.label_dict = self.corpus.make_label_dictionary()

        # Save trainer kwargs dict for reinitializations
        self.trainer_kwargs = kwargs

        # Load trainer with initial setup
        self._initial_setup(self.label_dict, **kwargs)

    def _initial_setup(self, label_dict: Dict, **kwargs):
        if self.predictive_head == "flair":

            # Get Document embeddings from `embeddings`
            document_embeddings: DocumentRNNEmbeddings = self.encoder.rnn_embeddings

            # Create the text classifier
            classifier = TextClassifier(
                document_embeddings, label_dictionary=label_dict, **kwargs,
            )

            # Initialize the text classifier trainer
            self.trainer = ModelTrainer(classifier, self.corpus)

        # TODO: In internal transformers package, create ****ForSequenceClassification adaptations
        elif self.predictive_head == "transformers":
            with open(self.encoder / "config.json") as config_f:
                configs = json.load(config_f)
                model_name = configs["architectures"][-1]
            if model_name == "BertForMaskedLM":
                pass

    def train(
        self,
        output_dir: Union[Path, str],
        learning_rate: float = 0.07,
        mini_batch_size: int = 32,
        anneal_factor: float = 0.5,
        patience: int = 5,
        max_epochs: int = 150,
        plot_weights: bool = False,
        **kwargs,
    ) -> None:
        """
        Train the Sequence Classifier

        * **output_dir** - The output directory where the model predictions and checkpoints will be written.
        * **learning_rate** - The initial learning rate
        * **mini_batch_size** - Batch size for the dataloader
        * **anneal_factor** - The factor by which the learning rate is annealed
        * **patience** - Patience is the number of epochs with no improvement the Trainer waits until annealing the learning rate
        * **max_epochs** - Maximum number of epochs to train. Terminates training if this number is surpassed.
        * **plot_weights** - Bool to plot weights or not
        * **kwargs** - Keyword arguments for the rest of Flair's `Trainer.train()` hyperparameters
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # Start the training
        self.trainer.train(
            output_dir,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            anneal_factor=anneal_factor,
            patience=patience,
            max_epochs=max_epochs,
            **kwargs,
        )

        # Plot weight traces
        if plot_weights:
            plotter = Plotter()
            plotter.plot_weights(output_dir / "weights.txt")

    def find_learning_rate(
        self,
        output_dir: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-8,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 32,
        stop_early: bool = True,
        smoothing_factor: float = 0.7,
        plot_learning_rate: bool = True,
        **kwargs,
    ) -> float:
        """
        Uses Leslie's cyclical learning rate finding method to generate and save the loss x learning rate plot

        This method returns a suggested learning rate using the static method `LMFineTuner.suggest_learning_rate()`
        which is implicitly run in this method.

        * **output_dir** - Path to dir for learning rate file to be saved
        * **file_name** - Name of learning rate .tsv file
        * **start_learning_rate** - Initial learning rate to start cyclical learning rate finder method
        * **end_learning_rate** - End learning rate to stop exponential increase of the learning rate
        * **iterations** - Number of optimizer iterations for the ExpAnnealLR scheduler
        * **mini_batch_size** - Batch size for dataloader
        * **stop_early** - Bool for stopping early once loss diverges
        * **smoothing_factor** - Smoothing factor on moving average of losses
        * **adam_epsilon** - Epsilon for Adam optimizer.
        * **weight_decay** - Weight decay if we apply some.
        * **kwargs** - Additional keyword arguments for the Adam optimizer
        **return** - Learning rate as a float
        """
        # 7. find learning rate
        learning_rate_tsv = self.trainer.find_learning_rate(
            base_path=output_dir,
            file_name=file_name,
            start_learning_rate=start_learning_rate,
            end_learning_rate=end_learning_rate,
            iterations=iterations,
            mini_batch_size=mini_batch_size,
            stop_early=stop_early,
            smoothing_factor=smoothing_factor,
        )

        # Reinitialize optimizer and parameters by reinitializing trainer
        self._initial_setup(self.label_dict, **self.trainer_kwargs)

        if plot_learning_rate:
            plotter = Plotter()
            plotter.plot_learning_rate(learning_rate_tsv)

        # Use the automated learning rate finder
        with open(learning_rate_tsv) as lr_f:
            lr_tsv = list(csv.reader(lr_f, delimiter="\t"))
        losses = np.array([float(row[-1]) for row in lr_tsv[1:]])
        lrs = np.array([float(row[-2]) for row in lr_tsv[1:]])
        lr_to_use = self.suggested_learning_rate(losses, lrs, **kwargs)
        print(f"Recommended Learning Rate {lr_to_use}")
        return lr_to_use

    @staticmethod
    def suggested_learning_rate(
        losses: np.array,
        lrs: np.array,
        lr_diff: int = 15,
        loss_threshold: float = 0.2,
        adjust_value: float = 1,
    ) -> float:
        # This seems redundant unless we can make this configured for each trainer/finetuner
        """
        Attempts to find the optimal learning rate using a interval slide rule approach with the cyclical learning rate method

        * **losses** - Numpy array of losses
        * **lrs** - Numpy array of exponentially increasing learning rates (must match dim of `losses`)
        * **lr_diff** - Learning rate Interval of slide ruler
        * **loss_threshold** - Threshold of loss difference on interval where the sliding stops
        * **adjust_value** - Coefficient for adjustment
        **return** - the optimal learning rate as a float
        """
        # Get loss values and their corresponding gradients, and get lr values
        assert lr_diff < len(losses)
        loss_grad = np.gradient(losses)

        # Search for index in gradients where loss is lowest before the loss spike
        # Initialize right and left idx using the lr_diff as a spacing unit
        # Set the local min lr as -1 to signify if threshold is too low
        r_idx = -1
        l_idx = r_idx - lr_diff
        local_min_lr = lrs[l_idx]
        while (l_idx >= -len(losses)) and (
            abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold
        ):
            local_min_lr = lrs[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * adjust_value

        return lr_to_use
