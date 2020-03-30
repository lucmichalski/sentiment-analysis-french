# Contains code used/modified by AdaptNLP author from transformers
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import collections
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
)

from adaptnlp.transformers.utils_squad import (
    SquadExample,
    InputFeatures,
    convert_examples_to_features,
    RawResult,
    RawResultExtended,
    get_final_text,
    _get_best_indexes,
    _compute_softmax,
)


class QuestionAnsweringModel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.config
        self.tokenizer
        self.model

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, query, context, top_n, as_dict):
        raise NotImplementedError


# TODO To be deprecated in the near future for a better module design
class BertQuestionAnsweringModel(QuestionAnsweringModel):
    def __init__(self):
        self.config = BertConfig
        self.tokenizer = BertTokenizer
        self.model = BertForQuestionAnswering
        self.model_names = list(self.config.pretrained_config_archive_map.keys())

        # Post Load
        self.pretrained_config = None
        self.pretrained_tokenizer = None
        self.pretrained_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_list(self, tensor: torch.Tensor) -> List[float]:
        return tensor.detach().cpu().tolist()

    def _load(self) -> None:
        print("Loading Pretrained Bert Question Answering Model...")
        model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        self.pretrained_config = self.config.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
        if "uncased" in model_name:
            tokenizer = self.tokenizer.from_pretrained(
                "bert-large-uncased", do_lower_case=True
            )
        else:
            tokenizer = self.tokenizer.from_pretrained(
                "bert-large-cased", do_lower_case=False
            )
        self.pretrained_tokenizer = tokenizer

        model = self.model.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=self.pretrained_config,
        )
        self.pretrained_model = model
        self.pretrained_model.to(self.device)

    def _load_one_query(
        self, query: str, context: str, output_examples=True
    ) -> Union[TensorDataset, List[SquadExample], List[InputFeatures]]:
        # Create doc_tokens for SquadExample with one query and context

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        # Create doc_tokens
        doc_tokens = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        # Create SquadExample
        examples = []
        example = SquadExample(
            qas_id=None,
            question_text=query,
            doc_tokens=doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False,
        )
        examples.append(example)

        # Convert to features
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self.pretrained_tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask,
        )

        if output_examples:
            return dataset, examples, features
        return dataset

    def _produce_concrete_predictions(
        self,
        all_examples,
        all_features,
        all_results,
        n_best_size=10,
        max_answer_length=30,
        do_lower_case=True,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
    ):

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True,
            )

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction",
                ["text", "start_index", "end_index", "start_logit", "end_logit"],
            )  # ### start_end_index

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                orig_doc_start = 0
                orig_doc_end = 0
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[
                        orig_doc_start : (orig_doc_end + 1)
                    ]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(
                        tok_text, orig_text, do_lower_case, verbose_logging
                    )
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        start_index=orig_doc_start,
                        end_index=orig_doc_end,
                    )
                )  # ### start_end_index...Make span indices inclusive
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(
                            text="",
                            start_logit=null_start_logit,
                            end_logit=null_end_logit,
                            start_index=0,
                            end_index=0,
                        )
                    )  # ### start_end_index should this be pred.<index>

                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(
                        0,
                        _NbestPrediction(
                            text="empty",
                            start_logit=0.0,
                            end_logit=0.0,
                            start_index=0.0,
                            end_index=0.0,
                        ),
                    )  # ### start_end_index

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(
                        text="empty",
                        start_logit=0.0,
                        end_logit=0.0,
                        start_index=0.0,
                        end_index=0.0,
                    )
                )  # ### start_end_index

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output[
                    "start_index"
                ] = (
                    entry.start_index
                )  # ### start_end_index MAGIC NUMBERS for adjustment :/
                output["end_index"] = entry.end_index
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = (
                    score_null
                    - best_non_null_entry.start_logit
                    - (best_non_null_entry.end_logit)
                )
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

            # All ids set as None so get rid of None Key
            all_predictions = all_predictions[None]
            all_nbest_json = all_nbest_json[None]

        return all_predictions, all_nbest_json

    def predict(
        self, query: str, context: str, n_best_size: int = 20
    ) -> Tuple[str, List[OrderedDict]]:
        """ Predicts top_n answer spans of query in regards to context

        Args:
            query: The question
            context: The context of which the question is asking
            top_n: The top n answers returned

        Returns:
            Either a list of string answers or a dict of the results
        """
        self._load() if not self.pretrained_model or not self.pretrained_tokenizer else None

        # Load and Evaluate Context Queries
        dataset, examples, features = self._load_one_query(query, context)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=5)
        all_results = []
        for batch in eval_dataloader:
            self.pretrained_model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                # BERT XLM XLNET DIFFERENCE
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]
                outputs = self.pretrained_model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                # BERT XLM XLNET DIFFERENCE
                result = RawResult(
                    unique_id=unique_id,
                    start_logits=self._to_list(outputs[0][i]),
                    end_logits=self._to_list(outputs[1][i]),
                )
                all_results.append(result)

        # Obtain Concrete Predictions
        all_predictions, all_nbest_json = self._produce_concrete_predictions(
            examples, features, all_results, n_best_size=n_best_size
        )
        return all_predictions, all_nbest_json


class XLNetQuestionAnsweringModel(QuestionAnsweringModel):
    def __init__(self):
        self.config = XLNetConfig
        self.tokenizer = XLNetTokenizer
        self.model = XLNetForQuestionAnswering
        self.model_names = list(self.config.pretrained_config_archive_map.keys())

        # Post Load
        self.pretrained_config = None
        self.pretrained_tokenizer = None
        self.pretrained_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_list(self, tensor: torch.Tensor) -> List[float]:
        return tensor.detach().cpu().tolist()

    def _load(self) -> None:
        print("Loading Pretrained XLNet Question Answering Model...")
        model_name = "xlnet-large-cased"
        self.pretrained_config = self.config.from_pretrained("xlnet-large-cased")
        tokenizer = self.tokenizer.from_pretrained(
            "xlnet-large-cased", do_lower_case=False
        )
        self.pretrained_tokenizer = tokenizer

        model = self.model.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=self.pretrained_config,
        )
        self.pretrained_model = model
        self.pretrained_model.to(self.device)

    def _load_one_query(
        self, query: str, context: str, output_examples=True
    ) -> Union[TensorDataset, List[SquadExample], List[InputFeatures]]:
        # Create doc_tokens for SquadExample with one query and context

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        # Create doc_tokens
        doc_tokens = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        # Create SquadExample
        examples = []
        example = SquadExample(
            qas_id=None,
            question_text=query,
            doc_tokens=doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False,
        )
        examples.append(example)

        # Convert to features
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self.pretrained_tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask,
        )

        if output_examples:
            return dataset, examples, features
        return dataset

    def _produce_concrete_predictions(
        self,
        all_examples,
        all_features,
        all_results,
        n_best_size=10,
        max_answer_length=30,
        verbose_logging=False,
    ):

        start_n_top = self.pretrained_model.config.start_n_top
        end_n_top = self.pretrained_model.config.end_n_top
        tokenizer = self.pretrained_tokenizer

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            [
                "feature_index",
                "start_index",
                "end_index",
                "start_log_prob",
                "end_log_prob",
            ],
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
        )

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive

            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]

                cur_null_score = result.cls_logits

                # if we could have irrelevant answers, get the min score of irrelevant
                score_null = min(score_null, cur_null_score)

                for i in range(start_n_top):
                    for j in range(end_n_top):
                        start_log_prob = result.start_top_log_probs[i]
                        start_index = result.start_top_index[i]

                        j_index = i * end_n_top + j

                        end_log_prob = result.end_top_log_probs[j_index]
                        end_index = result.end_top_index[j_index]

                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= feature.paragraph_len - 1:
                            continue
                        if end_index >= feature.paragraph_len - 1:
                            continue

                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob,
                            )
                        )

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_log_prob + x.end_log_prob),
                reverse=True,
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]

                # XLNet un-tokenizer
                # Let's keep it simple for now and see if we need all this later.
                #
                # tok_start_to_orig_index = feature.tok_start_to_orig_index
                # tok_end_to_orig_index = feature.tok_end_to_orig_index
                # start_orig_pos = tok_start_to_orig_index[pred.start_index]
                # end_orig_pos = tok_end_to_orig_index[pred.end_index]
                # paragraph_text = example.paragraph_text
                # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

                # Previously used Bert untokenizer
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, tokenizer.do_lower_case, verbose_logging
                )

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_log_prob=pred.start_log_prob,
                        end_log_prob=pred.end_log_prob,
                    )
                )

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6)
                )

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_log_prob + entry.end_log_prob)
                if not best_non_null_entry:
                    best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_log_prob"] = entry.start_log_prob
                output["end_log_prob"] = entry.end_log_prob
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            assert best_non_null_entry is not None

            score_diff = score_null
            scores_diff_json[example.qas_id] = score_diff
            # note(zhiliny): always predict best_non_null_entry
            # and the evaluation script will search for the best threshold
            all_predictions[example.qas_id] = best_non_null_entry.text

            all_nbest_json[example.qas_id] = nbest_json

        """
        if version_2_with_negative:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        with open(orig_data_file, "r", encoding='utf-8') as reader:
            orig_data = json.load(reader)["data"]
        """

        # All ids set as None so get rid of None Key
        all_predictions = all_predictions[None]
        all_nbest_json = all_nbest_json[None]

        return all_predictions, all_nbest_json

    def predict(
        self, query: str, context: str, n_best_size: int = 20, as_dict: bool = False
    ) -> Union[List[str], dict]:
        """ Predicts top_n answer spans of query in regards to context

        Args:
            query: The question
            context: The context of which the question is asking
            top_n: The top n answers returned
            as_dict: Returns answer in dict format if True

        Returns:
            Either a list of string answers or a dict of the results
        """
        self._load() if not self.pretrained_model or not self.pretrained_tokenizer else None

        # Load and Evaluate Context Queries
        dataset, examples, features = self._load_one_query(query, context)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=5)
        all_results = []
        for batch in eval_dataloader:
            self.pretrained_model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                # BERT XLM XLNET DIFFERENCE
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "cls_index": batch[4],
                    "p_mask": batch[5],
                }
                example_indices = batch[3]
                outputs = self.pretrained_model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                # BERT XLM XLNET DIFFERENCE
                result = RawResultExtended(
                    unique_id=unique_id,
                    start_top_log_probs=self._to_list(outputs[0][i]),
                    start_top_index=self._to_list(outputs[1][i]),
                    end_top_log_probs=self._to_list(outputs[2][i]),
                    end_top_index=self._to_list(outputs[3][i]),
                    cls_logits=self._to_list(outputs[4][i]),
                )
                all_results.append(result)

        # Obtain Concrete Predictions
        all_predictions, all_nbest_json = self._produce_concrete_predictions(
            examples, features, all_results, n_best_size=n_best_size
        )
        return all_predictions, all_nbest_json


class XLMQuestionAnsweringModel(QuestionAnsweringModel):
    def __init__(self):
        self.config = XLMConfig
        self.tokenizer = XLMTokenizer
        self.model = XLMForQuestionAnswering
        self.model_names = list(self.config.pretrained_config_archive_map.keys())

        # Post Load
        self.pretrained_config = None
        self.pretrained_tokenizer = None
        self.pretrained_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_list(self, tensor: torch.Tensor) -> List[float]:
        return tensor.detach().cpu().tolist()

    def _load(self) -> None:
        print("Loading Pretrained XLNet Question Answering Model...")
        model_name = "xlm-mlm-en-2048"
        self.pretrained_config = self.config.from_pretrained("xlm-mlm-en-2048")
        tokenizer = self.tokenizer.from_pretrained(
            "xlm-mlm-en-2048", do_lower_case=False
        )
        self.pretrained_tokenizer = tokenizer

        model = self.model.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=self.pretrained_config,
        )
        self.pretrained_model = model
        self.pretrained_model.to(self.device)

    def _load_one_query(
        self, query: str, context: str, output_examples=True
    ) -> Union[TensorDataset, List[SquadExample], List[InputFeatures]]:
        # Create doc_tokens for SquadExample with one query and context

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        # Create doc_tokens
        doc_tokens = []
        prev_is_whitespace = True
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        # Create SquadExample
        examples = []
        example = SquadExample(
            qas_id=None,
            question_text=query,
            doc_tokens=doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False,
        )
        examples.append(example)

        # Convert to features
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self.pretrained_tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask,
        )

        if output_examples:
            return dataset, examples, features
        return dataset

    def _produce_concrete_predictions(
        self,
        all_examples,
        all_features,
        all_results,
        n_best_size=10,
        max_answer_length=30,
        verbose_logging=False,
    ):

        start_n_top = self.pretrained_model.config.start_n_top
        end_n_top = self.pretrained_model.config.end_n_top
        tokenizer = self.pretrained_tokenizer

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            [
                "feature_index",
                "start_index",
                "end_index",
                "start_log_prob",
                "end_log_prob",
            ],
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
        )

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive

            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]

                cur_null_score = result.cls_logits

                # if we could have irrelevant answers, get the min score of irrelevant
                score_null = min(score_null, cur_null_score)

                for i in range(start_n_top):
                    for j in range(end_n_top):
                        start_log_prob = result.start_top_log_probs[i]
                        start_index = result.start_top_index[i]

                        j_index = i * end_n_top + j

                        end_log_prob = result.end_top_log_probs[j_index]
                        end_index = result.end_top_index[j_index]

                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= feature.paragraph_len - 1:
                            continue
                        if end_index >= feature.paragraph_len - 1:
                            continue

                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob,
                            )
                        )

            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_log_prob + x.end_log_prob),
                reverse=True,
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]

                # XLNet un-tokenizer
                # Let's keep it simple for now and see if we need all this later.
                #
                # tok_start_to_orig_index = feature.tok_start_to_orig_index
                # tok_end_to_orig_index = feature.tok_end_to_orig_index
                # start_orig_pos = tok_start_to_orig_index[pred.start_index]
                # end_orig_pos = tok_end_to_orig_index[pred.end_index]
                # paragraph_text = example.paragraph_text
                # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

                # Previously used Bert untokenizer
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text,
                    orig_text,
                    tokenizer,  # .do_lower_case, (a XLM problem?)
                    verbose_logging,
                )

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_log_prob=pred.start_log_prob,
                        end_log_prob=pred.end_log_prob,
                    )
                )

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6)
                )

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_log_prob + entry.end_log_prob)
                if not best_non_null_entry:
                    best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_log_prob"] = entry.start_log_prob
                output["end_log_prob"] = entry.end_log_prob
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            assert best_non_null_entry is not None

            score_diff = score_null
            scores_diff_json[example.qas_id] = score_diff
            # note(zhiliny): always predict best_non_null_entry
            # and the evaluation script will search for the best threshold
            all_predictions[example.qas_id] = best_non_null_entry.text

            all_nbest_json[example.qas_id] = nbest_json

        """
        if version_2_with_negative:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        with open(orig_data_file, "r", encoding='utf-8') as reader:
            orig_data = json.load(reader)["data"]
        """

        # All ids set as None so get rid of None Key
        all_predictions = all_predictions[None]
        all_nbest_json = all_nbest_json[None]

        return all_predictions, all_nbest_json

    def predict(
        self, query: str, context: str, n_best_size: int = 20, as_dict: bool = False
    ) -> Union[List[str], dict]:
        """ Predicts top_n answer spans of query in regards to context

        Args:
            query: The question
            context: The context of which the question is asking
            top_n: The top n answers returned
            as_dict: Returns answer in dict format if True

        Returns:
            Either a list of string answers or a dict of the results
        """
        self._load() if not self.pretrained_model or not self.pretrained_tokenizer else None

        # Load and Evaluate Context Queries
        dataset, examples, features = self._load_one_query(query, context)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=5)
        all_results = []
        for batch in eval_dataloader:
            self.pretrained_model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                # BERT XLM XLNET DIFFERENCE
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "cls_index": batch[4],
                    "p_mask": batch[5],
                }
                example_indices = batch[3]
                outputs = self.pretrained_model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                # BERT XLM XLNET DIFFERENCE
                result = RawResultExtended(
                    unique_id=unique_id,
                    start_top_log_probs=self._to_list(outputs[0][i]),
                    start_top_index=self._to_list(outputs[1][i]),
                    end_top_log_probs=self._to_list(outputs[2][i]),
                    end_top_index=self._to_list(outputs[3][i]),
                    cls_logits=self._to_list(outputs[4][i]),
                )
                all_results.append(result)

        # Obtain Concrete Predictions
        all_predictions, all_nbest_json = self._produce_concrete_predictions(
            examples, features, all_results, n_best_size=n_best_size
        )
        return all_predictions, all_nbest_json
