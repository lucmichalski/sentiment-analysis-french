import logging
from typing import Tuple, List, Union, Dict
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    XLNetForQuestionAnswering,
    XLMForQuestionAnswering,
    CamembertForQuestionAnswering,
    DistilBertForQuestionAnswering,
    RobertaForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    SquadExample,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult

from adaptnlp.transformers import BertQuestionAnsweringModel
from adaptnlp.model import AdaptiveModel
from adaptnlp.transformers.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
)

logger = logging.getLogger(__name__)


class TransformersQuestionAnswering(AdaptiveModel):
    """ Adaptive Model for Transformers Question Answering Model

    **Parameters**

    * **tokenizer** - A tokenizer object from Huggingface's transformers (TODO)and tokenizers *
    * **model** - A transformer Question Answering model

    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        # Load up model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

        # Setup cuda and automatic allocation of model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.xmodel_instances = (XLNetForQuestionAnswering, XLMForQuestionAnswering)

    @classmethod
    def load(cls, model_name_or_path: str) -> AdaptiveModel:
        """ Class method for loading and constructing this model

        * **model_name_or_path** - A key string of one of Transformer's pre-trained Question Answering (SQUAD) models
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        qa_model = cls(tokenizer, model)
        return qa_model

    def predict(
        self,
        query: Union[List[str], str],
        context: Union[List[str], str],
        n_best_size: int = 5,
        mini_batch_size: int = 32,
        max_answer_length: int = 10,
        do_lower_case: bool = False,
        version_2_with_negative: bool = False,
        verbose_logging: bool = False,
        null_score_diff_threshold: float = 0.0,
        max_seq_length: int = 512,
        doc_stride: int = 128,
        max_query_length: int = 64,
        **kwargs,
    ) -> Tuple[Tuple[str, List[OrderedDict]], Tuple[OrderedDict, OrderedDict]]:
        """ Predict method for running inference using the pre-trained question answering model

        * **query** - String or list of strings that specify the ordered questions corresponding to `context`
        * **context** - String or list of strings that specify the ordered contexts corresponding to `query`
        * **n_best_size** - Number of top n results you want
        * **mini_batch_size** - Mini batch size
        * **max_answer_length** - Maximum token length for answers that are returned
        * **do_lower_case** - Set as `True` if using uncased QA models
        * **version_2_with_negative** - Set as True if using QA model with SQUAD2.0
        * **verbose_logging** - Set True if you want prediction verbose loggings
        * **null_score_diff_threshold** - Threshold for predicting null(no answer) in Squad 2.0 Model.  Default is 0.0.  Raise this if you want fewer null answers
        * **max_seq_length** - Maximum context token length. Check model configs to see max sequence length the model was trained with
        * **doc_stride** - Number of token strides to take when splitting up conext into chunks of size `max_seq_length`
        * **max_query_length** - Maximum token length for queries
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers model (mostly for saving evaluations)
        """
        # Make string input consistent as list
        if isinstance(query, str):
            query = [query]
            context = [context]
        assert len(query) == len(context)

        examples = self._mini_squad_processor(query=query, context=context)
        features, dataset = squad_convert_examples_to_features(
            examples,
            self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )
        all_results = []

        with torch.no_grad():

            dataloader = DataLoader(dataset, batch_size=mini_batch_size)

            for batch in tqdm(dataloader, desc="Predicting answer"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]

                if isinstance(
                    self.model,
                    (
                        XLMForQuestionAnswering,
                        RobertaForQuestionAnswering,
                        DistilBertForQuestionAnswering,
                        CamembertForQuestionAnswering,
                    ),
                ):
                    del inputs["token_type_ids"]

                # XLNet and XLM use more arguments for their predictions
                if isinstance(self.model, self.xmodel_instances):
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(self.model, "config") and hasattr(
                        self.model.config, "lang2id"
                    ):
                        # Set language id as 0 for now
                        inputs.update(
                            {
                                "langs": (
                                    torch.ones(batch[0].shape, dtype=torch.int64) * 0
                                ).to(self.device)
                            }
                        )

                outputs = self.model(**inputs)

                # Iterate through and produce `SquadResults
                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)

                    output = [self.to_list(output[i]) for output in outputs]

                    if isinstance(self.model, self.xmodel_instances):
                        # Some models like the ones in `self.xmodel_instances` use 5 arguments for their predictions
                        start_logits = output[0]
                        start_top_index = output[1]
                        end_logits = output[2]
                        end_top_index = output[3]
                        cls_logits = output[4]

                        result = SquadResult(
                            unique_id,
                            start_logits,
                            end_logits,
                            start_top_index=start_top_index,
                            end_top_index=end_top_index,
                            cls_logits=cls_logits,
                        )

                    else:
                        start_logits, end_logits = output
                        result = SquadResult(unique_id, start_logits, end_logits)
                    all_results.append(result)

            if isinstance(self.model, self.xmodel_instances):
                start_n_top = (
                    self.model.config.start_n_top
                    if hasattr(self.model, "config")
                    else self.model.module.config.start_n_top
                )
                end_n_top = (
                    self.model.config.end_n_top
                    if hasattr(self.model, "config")
                    else self.model.module.config.end_n_top
                )

                answers, n_best = compute_predictions_log_probs(
                    examples,
                    features,
                    all_results,
                    n_best_size,
                    max_answer_length,
                    start_n_top,
                    end_n_top,
                    version_2_with_negative,
                    self.tokenizer,
                    verbose_logging,
                    **kwargs,
                )

            else:
                answers, n_best = compute_predictions_logits(
                    examples,
                    features,
                    all_results,
                    n_best_size,
                    max_answer_length,
                    do_lower_case,
                    verbose_logging,
                    version_2_with_negative,
                    null_score_diff_threshold,
                    self.tokenizer,
                    **kwargs,
                )

        return answers, n_best

    def to_list(self, tensor: torch.Tensor):
        return tensor.detach().cpu().tolist()

    def _mini_squad_processor(
        self, query: List[str], context: List[str]
    ) -> List[SquadExample]:
        """ Squad data processor to create `SquadExamples`

        * **query** - List of query strings, must be same length as `context`
        * **context** - List of context strings, must be same length as `query`

        """
        assert len(query) == len(context)
        examples = []
        title = "qa"
        is_impossible = False
        answer_text = None
        start_position_character = None
        answers = ["answer"]
        for idx, (q, c) in enumerate(zip(query, context)):
            example = SquadExample(
                qas_id=str(idx),
                question_text=q,
                context_text=c,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )
            examples.append(example)
        return examples


class EasyQuestionAnswering:
    """Question Answering Module

    Usage:

    ```python
    >>> qa = adaptnlp.EasyQuestionAnswering()
    >>> qa.predict_qa(query="What is life?", context="Life is NLP.", n_best_size=5, mini_batch_size=1)
    ```
    """

    def __init__(self):
        self.models: Dict[AdaptiveModel] = defaultdict(bool)

        # Soon to deprecated
        self.bert_qa = None

    # Dynamic Loaders
    def _load_bert_qa(self) -> None:
        self.bert_qa = BertQuestionAnsweringModel()

    # Soon to be deprecated
    def predict_bert_qa(
        self, query: str, context: str, n_best_size: int = 20
    ) -> Tuple[str, List[OrderedDict]]:
        """ Predicts top_n answer spans of query in regards to context


        * **query** - The question
        * **context** - The context of which the question is asking
        * **n_best_size** - The top n answers returned

        **return** - Either a list of string answers or a dict of the results
        """

        self._load_bert_qa() if not self.bert_qa else None
        return self.bert_qa.predict(
            query=query, context=context, n_best_size=n_best_size
        )

    def predict_qa(
        self,
        query: Union[List[str], str],
        context: Union[List[str], str],
        n_best_size: int = 5,
        mini_batch_size: int = 32,
        model_name_or_path: str = "bert-large-uncased-whole-word-masking-finetuned-squad",
        **kwargs,
    ) -> Tuple[Tuple[str, List[OrderedDict]], Tuple[OrderedDict, OrderedDict]]:
        """ Predicts top_n answer spans of query in regards to context

        * **query** - String or list of strings that specify the ordered questions corresponding to `context`
        * **context** - String or list of strings that specify the ordered contexts corresponding to `query`
        * **n_best_size** - The top n answers returned
        * **mini_batch_size** - Mini batch size for inference
        * **model_name_or_path** - Path to QA model or name of QA model at huggingface.co/models
        * **kwargs**(Optional) - Keyword arguments for `AdaptiveModel`s like `TransformersQuestionAnswering`

        **return** - Either a list of string answers or a dict of the results
        """
        try:
            if not self.models[model_name_or_path]:
                self.models[model_name_or_path] = TransformersQuestionAnswering.load(
                    model_name_or_path
                )
        except OSError:
            logger.info(
                f"{model_name_or_path} not a valid Transformers pre-trained QA model...check path or huggingface.co/models"
            )
            raise ValueError(
                f"{model_name_or_path} is not a valid path or model name from huggingface.co/models"
            )
            return OrderedDict(), [OrderedDict()]

        model = self.models[model_name_or_path]

        # If query and context is just one instance, get rid of nested OrderedDict
        if isinstance(query, str):
            top_answer, top_n_answers = model.predict(
                query=query,
                context=context,
                n_best_size=n_best_size,
                mini_batch_size=mini_batch_size,
                **kwargs,
            )
            return top_answer["0"], top_n_answers["0"]

        return model.predict(
            query=query,
            context=context,
            n_best_size=n_best_size,
            mini_batch_size=mini_batch_size,
            **kwargs,
        )
