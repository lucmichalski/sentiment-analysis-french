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
from typing import Union, List
import datetime
import glob
import logging
import os
import pickle
import random
import re
import shutil
import csv
import copy
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import trange
from tqdm import tqdm as tqdm_base

from flair.visual.training_curves import Plotter

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertTokenizer,
    get_linear_schedule_with_warmup,
)


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, "_instances"):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        overwrite_cache: bool,
        file_path: str = "train",
        block_size: int = 512,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - (
            tokenizer.max_len - tokenizer.max_len_single_sentence
        )

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            print("Opening file")
            with open(file_path, encoding="utf-8") as f:
                print(f"Reading file {file_path}")
                if file_path.endswith(".txt") or file_path.endswith(".raw"):
                    text = f.read()
                elif file_path.endswith(".csv"):
                    reader = csv.reader(f)
                    try:
                        csv_idx = next(reader).index("text")
                        text = ". ".join([row[csv_idx] for row in reader])
                    except ValueError:
                        logger.info("No header row provided with 'text' column")
                        text = ""
                else:
                    text = ""
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            print("iterating through tokenized text")
            for i in tqdm(
                range(0, len(tokenized_text) - block_size + 1, block_size)
            ):  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                )
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


class LMFineTuner:
    """
     A Language Model Fine Tuner object you can set language model configurations and then train and evaluate

    Usage:

    ```python
    >>> finetuner = adaptnlp.LMFineTuner()
    >>> finetuner.train()
    ```

    **Parameters:**

    * **train_data_file** - The input training data file (a text file).
    * **eval_data_file** - An optional input evaluation data file to evaluate the perplexity on (a text file).
    * **model_type** - The model architecture to be trained or fine-tuned.
    * **model_name_or_path** - The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
    * **mlm** - Train with masked-language modeling loss instead of language modeling.
    * **mlm_probability** - Ratio of tokens to mask for masked language modeling loss
    * **config_name** - Optional Transformers pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.
    * **tokenizer_name** - Optional Transformers pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.
    * **cache_dir** - Optional directory to store the pre-trained models downloaded from s3 (If None, will go to default dir)
    * **block_size** - Optional input sequence length after tokenization.
                        The training dataset will be truncated in block of this size for training."
                        `-1` will default to the model max input length for single sentence inputs (take into account special tokens).
    * **no_cuda** - Avoid using CUDA when available
    * **overwrite_cache** - Overwrite the cached training and evaluation sets
    * **seed** - random seed for initialization
    * **fp16** - Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
    * **fp16_opt_level** - For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
    * **local_rank** - For distributed training: local_rank
    """

    def __init__(
        self,
        train_data_file: str,
        eval_data_file: str = None,
        model_type: str = "bert",
        model_name_or_path: str = None,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        config_name: str = None,
        tokenizer_name: str = None,
        cache_dir: str = None,
        block_size: int = -1,
        no_cuda: bool = False,
        overwrite_cache: bool = False,
        seed: int = 42,
        fp16: bool = False,
        fp16_opt_level: str = "O1",
        local_rank: int = -1,
    ):

        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.block_size = block_size
        self.no_cuda = no_cuda
        self.overwrite_cache = overwrite_cache
        self.seed = seed
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.local_rank = local_rank

        self.MODEL_CLASSES = {
            "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
            "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
            "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
            "distilbert": (
                DistilBertConfig,
                DistilBertForMaskedLM,
                DistilBertTokenizer,
            ),
            "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
            "albert": (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
        }

        self._initial_setup()

    def _initial_setup(self):

        #  Setup model type and output directory
        if (
            self.model_type in ["bert", "roberta", "distilbert", "camembert"]
            and not self.mlm
        ):
            raise ValueError(
                "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using with the mlm parameter set as `True`"
                "for (masked language modeling)."
            )

        if self.eval_data_file is None:
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a file to eval_data_file parameter or continue without evaluating"
            )

        #  Setup CUDA, GPU, and distributed training
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
            )
            self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        self.device = device

        # Setup logging and seed
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.local_rank,
            device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )
        self._set_seed()

        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        config_class, model_class, tokenizer_class = self.MODEL_CLASSES[self.model_type]
        if self.config_name:
            self.config = config_class.from_pretrained(
                self.config_name, cache_dir=self.cache_dir
            )
        elif self.model_name_or_path:
            self.config = config_class.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir
            )
        else:
            self.config = config_class()

        if self.tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.tokenizer_name, cache_dir=self.cache_dir
            )
        elif self.model_name_or_path:
            self.tokenizer = tokenizer_class.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir
            )
        else:
            raise ValueError(
                f"You are instantiating a new {tokenizer_class.__name__} tokenizer from scratch. Are you sure this is what you meant to do? \n To specifiy a pretrained tokenizer name, pass in a tokenizer_name argument"
            )

        if self.block_size <= 0:
            self.block_size = self.tokenizer.max_len_single_sentence
            # Our input block size will be the max possible for the model
        else:
            self.block_size = min(
                self.block_size, self.tokenizer.max_len_single_sentence
            )

        if self.model_name_or_path:
            self.model = model_class.from_pretrained(
                self.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_name_or_path),
                config=self.config,
                cache_dir=self.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            self.model = model_class(config=self.config)

        self.model.to(self.device)

        if self.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    def train_one_cycle(
        self,
        output_dir: str,
        should_continue: bool = False,
        overwrite_output_dir: bool = False,
        evaluate_during_training: bool = False,
        per_gpu_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        num_train_epochs: float = 1.0,
        max_steps: int = -1,
        warmup_steps: int = 0,
        logging_steps: int = 50,
        save_steps: int = 50,
        save_total_limit: int = 3,
        use_tensorboard: bool = False,
    ) -> None:
        """

        * **output_dir** - The output directory where the model predictions and checkpoints will be written.
        * **should_continue** - Whether to continue training from latest checkpoint in `output_dir`
        * **overwrite_output_dir** - Overwrite the content of output directory `output_dir`
        * **evaluate_during_training** - Run evaluation during training at each `logging_step`.
        * **per_gpu_train_batch_size** - Batch size per GPU/CPU for training. (If `evaluate_during_training` is True, this is also the eval batch size
        * **gradient_accumulation_steps** - Number of updates steps to accumulate before performing a backward/update pass
        * **learning_rate** - The initial learning rate for Adam optimizer.
        * **weight_decay** - Weight decay if we apply some.
        * **adam_epsilon** - Epsilon for Adam optimizer.
        * **max_grad_norm** - Max gradient norm. Duh
        * **num_train_epochs** - Total number of training epochs to perform.
        * **max_steps** - If > 0: set total number of training steps to perform. Override `num_train_epochs`.
        * **warmup_steps** - Linear warmup over warmup_steps.
        * **logging_steps** - Number of steps until logging occurs.
        * **save_steps** - Number of steps until checkpoint is saved in `output_dir`
        * **save_total_limit** - Limit the total amount of checkpoints, delete the older checkpoints in the `output_dir`, does not delete by default
        * **use_tensorboard** - Only useable if tensorboard is installed
        **return** - None
        """
        # Check to overwrite
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. Set overwrite_output_dir as `True` to overcome."
            )

        # Check if continuing training from checkpoint
        if should_continue:
            sorted_checkpoints = self._sorted_checkpoints(
                checkpoint_prefix="checkpoint", output_dir=output_dir
            )
            if len(sorted_checkpoints) == 0:
                raise ValueError(
                    f"Trying to continue training but no checkpoint was found in {output_dir}, set `should_continue` argument to False if training from scratch"
                )
            else:
                self.model_name_or_path = sorted_checkpoints[-1]
                self._initial_setup()

        # Get locals for training args
        init_locals = copy.deepcopy(locals())
        init_locals.pop("self")

        # Start logger
        logger.info("Training/evaluation parameters %s", str(locals()))

        ##############
        ## Training ##
        ##############
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load Dataset
        train_dataset = self.load_and_cache_examples(evaluate=False)

        if self.local_rank == 0:
            torch.distributed.barrier()

        if self.local_rank in [-1, 0] and use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter()
            except ImportError:
                logger.warning(
                    "WARNING! Tensorboard is a required dependency...`use_tensorboard` is now set as False"
                )
                use_tensorboard = False
                pass

        # Train the model

        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        train_sampler = (
            RandomSampler(train_dataset)
            if self.local_rank == -1
            else DistributedSampler(train_dataset)
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=train_batch_size,
            collate_fn=collate,
        )

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = (
                max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
            )
        else:
            t_total = (
                len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=int(num_train_epochs),
            steps_per_epoch=int(t_total),
        )

        # Check if saved optimizer or scheduler states exist
        if (
            self.model_name_or_path
            and os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(self.model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "scheduler.pt"))
            )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.fp16_opt_level
            )

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // gradient_accumulation_steps
                )

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0

        model_to_resize = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(self.tokenizer))

        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(num_train_epochs),
            desc="Epoch",
            disable=self.local_rank not in [-1, 0],
        )
        self._set_seed()  # Added here for reproducibility
        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=self.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = self.mask_tokens(batch) if self.mlm else (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                outputs = (
                    self.model(inputs, masked_lm_labels=labels)
                    if self.mlm
                    else self.model(inputs, labels=labels)
                )
                loss = outputs[
                    0
                ]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.local_rank in [-1, 0]
                        and logging_steps > 0
                        and global_step % logging_steps == 0
                    ):
                        # Log metrics
                        if (
                            self.local_rank == -1 and evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(
                                output_dir=output_dir,
                                per_gpu_eval_batch_size=per_gpu_train_batch_size,
                            )
                            if use_tensorboard:
                                for key, value in results.items():
                                    tb_writer.add_scalar(
                                        f"eval_{key}", value, global_step
                                    )
                        if use_tensorboard:
                            tb_writer.add_scalar(
                                "lr", scheduler.get_lr()[0], global_step
                            )
                            tb_writer.add_scalar(
                                "loss",
                                (tr_loss - logging_loss) / logging_steps,
                                global_step,
                            )
                        logging_loss = tr_loss

                    if (
                        self.local_rank in [-1, 0]
                        and save_steps > 0
                        and global_step % save_steps == 0
                    ):
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        # TODO: os.makedirs bug when output_dir exists
                        ckpt_output_dir = os.path.join(
                            output_dir, f"{checkpoint_prefix}-{global_step}"
                        )
                        os.makedirs(ckpt_output_dir, exist_ok=True)
                        model_to_save = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(ckpt_output_dir)
                        self.tokenizer.save_pretrained(ckpt_output_dir)

                        torch.save(
                            init_locals,
                            os.path.join(ckpt_output_dir, "training_args.bin"),
                        )
                        logger.info("Saving model checkpoint to %s", ckpt_output_dir)

                        self._rotate_checkpoints(
                            checkpoint_prefix,
                            save_total_limit=save_total_limit,
                            output_dir=output_dir,
                        )

                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(ckpt_output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(ckpt_output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s",
                            ckpt_output_dir,
                        )

                if max_steps > 0 and global_step > max_steps:
                    epoch_iterator.close()
                    break
            if max_steps > 0 and global_step > max_steps:
                train_iterator.close()
                break

        if self.local_rank in [-1, 0] and use_tensorboard:
            tb_writer.close()

        tr_loss = tr_loss / global_step

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if self.local_rank in [-1, 0]:
                os.makedirs(output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(init_locals, os.path.join(output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            config_class, model_class, tokenizer_class = self.MODEL_CLASSES[
                self.model_type
            ]
            self.model = model_class.from_pretrained(output_dir)
            self.tokenizer = tokenizer_class.from_pretrained(output_dir)
            self.model.to(self.device)

    def train(
        self,
        output_dir: str,
        should_continue: bool = False,
        overwrite_output_dir: bool = False,
        evaluate_during_training: bool = False,
        per_gpu_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        num_train_epochs: float = 1.0,
        max_steps: int = -1,
        warmup_steps: int = 0,
        logging_steps: int = 50,
        save_steps: int = 50,
        save_total_limit: int = 3,
        use_tensorboard: bool = False,
    ) -> None:
        """

        * **output_dir** - The output directory where the model predictions and checkpoints will be written.
        * **should_continue** - Whether to continue training from latest checkpoint in `output_dir`
        * **overwrite_output_dir** - Overwrite the content of output directory `output_dir`
        * **evaluate_during_training** - Run evaluation during training at each `logging_step`.
        * **per_gpu_train_batch_size** - Batch size per GPU/CPU for training. (If `evaluate_during_training` is True, this is also the eval batch size
        * **gradient_accumulation_steps** - Number of updates steps to accumulate before performing a backward/update pass
        * **learning_rate** - The initial learning rate for Adam optimizer.
        * **weight_decay** - Weight decay if we apply some.
        * **adam_epsilon** - Epsilon for Adam optimizer.
        * **max_grad_norm** - Max gradient norm. Duh
        * **num_train_epochs** - Total number of training epochs to perform.
        * **max_steps** - If > 0: set total number of training steps to perform. Override `num_train_epochs`.
        * **warmup_steps** - Linear warmup over warmup_steps.
        * **logging_steps** - Number of steps until logging occurs.
        * **save_steps** - Number of steps until checkpoint is saved in `output_dir`
        * **save_total_limit** - Limit the total amount of checkpoints, delete the older checkpoints in the `output_dir`, does not delete by default
        * **use_tensorboard** - Only useable if tensorboard is installed
        **return** - None
        """
        # Check to overwrite
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. Set overwrite_output_dir as `True` to overcome."
            )

        # Check if continuing training from checkpoint
        if should_continue:
            sorted_checkpoints = self._sorted_checkpoints(
                checkpoint_prefix="checkpoint", output_dir=output_dir
            )
            if len(sorted_checkpoints) == 0:
                raise ValueError(
                    f"Trying to continue training but no checkpoint was found in {output_dir}, set `should_continue` argument to False if training from scratch"
                )
            else:
                self.model_name_or_path = sorted_checkpoints[-1]
                self._initial_setup()

        # Get locals for training args
        init_locals = copy.deepcopy(locals())
        init_locals.pop("self")

        # Start logger
        logger.info("Training/evaluation parameters %s", str(locals()))

        ##############
        ## Training ##
        ##############
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load Dataset
        train_dataset = self.load_and_cache_examples(evaluate=False)

        if self.local_rank == 0:
            torch.distributed.barrier()

        if self.local_rank in [-1, 0] and use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter()
            except ImportError:
                logger.warning(
                    "WARNING! Tensorboard is a required dependency...`use_tensorboard` is now set as False"
                )
                use_tensorboard = False
                pass

        # Train the model

        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        train_sampler = (
            RandomSampler(train_dataset)
            if self.local_rank == -1
            else DistributedSampler(train_dataset)
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=train_batch_size,
            collate_fn=collate,
        )

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = (
                max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
            )
        else:
            t_total = (
                len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
            self.model_name_or_path
            and os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(self.model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "scheduler.pt"))
            )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.fp16_opt_level
            )

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // gradient_accumulation_steps
                )

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0

        model_to_resize = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(self.tokenizer))

        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(num_train_epochs),
            desc="Epoch",
            disable=self.local_rank not in [-1, 0],
        )
        self._set_seed()  # Added here for reproducibility
        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=self.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = self.mask_tokens(batch) if self.mlm else (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                outputs = (
                    self.model(inputs, masked_lm_labels=labels)
                    if self.mlm
                    else self.model(inputs, labels=labels)
                )
                loss = outputs[
                    0
                ]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.local_rank in [-1, 0]
                        and logging_steps > 0
                        and global_step % logging_steps == 0
                    ):
                        # Log metrics
                        if (
                            self.local_rank == -1 and evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(
                                output_dir=output_dir,
                                per_gpu_eval_batch_size=per_gpu_train_batch_size,
                            )
                            if use_tensorboard:
                                for key, value in results.items():
                                    tb_writer.add_scalar(
                                        f"eval_{key}", value, global_step
                                    )
                        if use_tensorboard:
                            tb_writer.add_scalar(
                                "lr", scheduler.get_lr()[0], global_step
                            )
                            tb_writer.add_scalar(
                                "loss",
                                (tr_loss - logging_loss) / logging_steps,
                                global_step,
                            )
                        logging_loss = tr_loss

                    if (
                        self.local_rank in [-1, 0]
                        and save_steps > 0
                        and global_step % save_steps == 0
                    ):
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        # TODO: os.makedirs bug when output_dir exists
                        ckpt_output_dir = os.path.join(
                            output_dir, f"{checkpoint_prefix}-{global_step}"
                        )
                        os.makedirs(ckpt_output_dir, exist_ok=True)
                        model_to_save = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(ckpt_output_dir)
                        self.tokenizer.save_pretrained(ckpt_output_dir)

                        torch.save(
                            init_locals,
                            os.path.join(ckpt_output_dir, "training_args.bin"),
                        )
                        logger.info("Saving model checkpoint to %s", ckpt_output_dir)

                        self._rotate_checkpoints(
                            checkpoint_prefix,
                            save_total_limit=save_total_limit,
                            output_dir=output_dir,
                        )

                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(ckpt_output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(ckpt_output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s",
                            ckpt_output_dir,
                        )

                if max_steps > 0 and global_step > max_steps:
                    epoch_iterator.close()
                    break
            if max_steps > 0 and global_step > max_steps:
                train_iterator.close()
                break

        if self.local_rank in [-1, 0] and use_tensorboard:
            tb_writer.close()

        tr_loss = tr_loss / global_step

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if self.local_rank in [-1, 0]:
                os.makedirs(output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(init_locals, os.path.join(output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            config_class, model_class, tokenizer_class = self.MODEL_CLASSES[
                self.model_type
            ]
            self.model = model_class.from_pretrained(output_dir)
            self.tokenizer = tokenizer_class.from_pretrained(output_dir)
            self.model.to(self.device)

    def evaluate_all_checkpoints(
        self, output_dir: str, per_gpu_eval_batch_size: int = 4,
    ) -> dict:
        """
        * **output_dir** - The output directory where the model predictions and checkpoints will be written.
        * **per_gpu_eval_batch_size** - Batch size per GPU/CPU for evaluation.
        **return** - Results in a dictionary
        """
        # Evaluation
        results = {}
        if self.local_rank in [-1, 0]:
            # checkpoints = [output_dir]
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            print(checkpoints)
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = (
                    checkpoint.split("/")[-1]
                    if checkpoint.find("checkpoint") != -1
                    else ""
                )

                config_class, model_class, tokenizer_class = self.MODEL_CLASSES[
                    self.model_type
                ]
                self.model = model_class.from_pretrained(checkpoint)
                self.model.to(self.device)
                result = self.evaluate(
                    output_dir=output_dir,
                    per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                    prefix=prefix,
                )
                result = dict((k + f"_{global_step}", v) for k, v in result.items())
                results.update(result)
        return results

    def evaluate(
        self, output_dir: str, per_gpu_eval_batch_size: int = 4, prefix: str = "",
    ) -> dict:
        """
        * **output_dir** - The output directory where the model predictions and checkpoints will be written.
        * **per_gpu_eval_batch_size** - Batch size per GPU/CPU for evaluation.
        * **prefix** - Prefix of checkpoint i.e. "checkpoint-50"
        **return** - Results in a dictionary
        """
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = output_dir

        eval_dataset = self.load_and_cache_examples(evaluate=True)

        if self.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        eval_batch_size = per_gpu_eval_batch_size * max(1, self.n_gpu)

        # Note that DistributedSampler samples randomly
        def collate(examples: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=eval_batch_size,
            collate_fn=collate,
        )

        # multi-gpu evaluate
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = self.mask_tokens(batch) if self.mlm else (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = (
                    self.model(inputs, masked_lm_labels=labels)
                    if self.mlm
                    else self.model(inputs, labels=labels)
                )
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Eval results {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    def load_and_cache_examples(self, evaluate=False):
        return TextDataset(
            self.tokenizer,
            self.model_type,
            self.overwrite_cache,
            file_path=self.eval_data_file if evaluate else self.train_data_file,
            block_size=self.block_size,
        )

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def _sorted_checkpoints(
        self, checkpoint_prefix: str, output_dir: str, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = glob.glob(os.path.join(output_dir, f"{checkpoint_prefix}-*"))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path)
                    )

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(
        self,
        checkpoint_prefix: str,
        save_total_limit: int,
        output_dir: str,
        use_mtime: bool = False,
    ) -> None:
        if not save_total_limit:
            return
        if save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            checkpoint_prefix, output_dir, use_mtime
        )
        if len(checkpoints_sorted) <= save_total_limit:
            return
        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - save_total_limit
        )
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] due to save_total_limit"
            )
            shutil.rmtree(checkpoint)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Set `mlm` param as False"
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def find_learning_rate(
        self,
        output_dir: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-7,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 8,
        stop_early: bool = True,
        smoothing_factor: float = 0.7,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
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
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        if type(output_dir) is str:
            output_dir = Path(output_dir)
        from flair.training_utils import (
            init_output_file,
            log_line,
        )

        learning_rate_tsv = init_output_file(output_dir, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        # Prepare optimizer and schedule (linear warmup and decay) for transformer's AdamW optimzer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=start_learning_rate,
            eps=adam_epsilon,
            **kwargs,
        )
        ## Original SGD optimizer Flair uses commnted out
        # optimizer = AdamW(self.model.parameters(), lr=start_learning_rate, **kwargs)
        # from torch.optim.sgd import SGD
        # optimizer = SGD(
        #    self.model.parameters(), lr=start_learning_rate, **kwargs
        # )

        # Flair's original EXPAnnealLR scheduler
        from flair.optim import ExpAnnealLR

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        self.model.train()

        train_dataset = self.load_and_cache_examples(evaluate=False)

        step = 0
        while step < iterations:
            train_sampler = RandomSampler(train_dataset)
            batch_loader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=mini_batch_size
            )
            for batch in batch_loader:
                step += 1

                # forward pass
                inputs, labels = self.mask_tokens(batch) if self.mlm else (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                outputs = (
                    self.model(inputs, masked_lm_labels=labels)
                    if self.mlm
                    else self.model(inputs, labels=labels)
                )
                loss = outputs[0]

                # update optimizer and scheduler
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step(step)

                print(scheduler.get_lr())
                learning_rate = scheduler.get_lr()[0]

                loss_item = loss.item()
                if step == 1:
                    best_loss = loss_item
                else:
                    if smoothing_factor > 0:
                        moving_avg_loss = (
                            smoothing_factor * moving_avg_loss
                            + (1 - smoothing_factor) * loss_item
                        )
                        loss_item = moving_avg_loss / (
                            1 - smoothing_factor ** (step + 1)
                        )
                    if loss_item < best_loss:
                        best_loss = loss

                if step > iterations:
                    break

                if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                    log_line(logger)
                    logger.info("loss diverged - stopping early!")
                    step = iterations
                    break

                with open(str(learning_rate_tsv), "a") as f:
                    f.write(
                        f"{step}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
                    )

            self.model.load_state_dict(model_state)
            self.model.to(self.device)

        log_line(logger)
        logger.info(
            f"learning rate finder finished - plot {learning_rate_tsv} \n Reinitalizing model's parameters and optimizer"
        )
        log_line(logger)

        # Reinitialize transformers model's parameters (This could be optimized)
        self._initial_setup()

        plotter = Plotter()
        plotter.plot_learning_rate(Path(learning_rate_tsv))

        # Use the automated learning rate finder
        with open(learning_rate_tsv) as lr_f:
            lr_tsv = list(csv.reader(lr_f, delimiter="\t"))
        losses = np.array([float(row[-1]) for row in lr_tsv[1:]])
        lrs = np.array([float(row[-2]) for row in lr_tsv[1:]])
        lr_to_use = self.suggest_learning_rate(losses, lrs)
        print(f"Recommended Learning Rate {lr_to_use}")
        return lr_to_use

    @staticmethod
    def suggest_learning_rate(
        losses: np.array,
        lrs: np.array,
        lr_diff: int = 30,
        loss_threshold: float = 0.05,
        adjust_value: float = 1,
    ) -> float:
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

    def freeze_to(self, n: int) -> None:
        """Freeze first n layers of model

        * **n** - Starting from initial layer, freeze all layers up to nth layer inclusively
        """
        layers = list(self.model.parameters())
        # Freeze up to n layers
        for param in layers[:n]:
            param.requires_grad = False
        for param in layers[n:]:
            param.requires_grad = True

    def freeze(self) -> None:
        """Freeze last classification layer group only
        """
        layers_len = len(list(self.model.cls.parameters()))
        self.freeze_to(-layers_len)

    def unfreeze(self) -> None:
        """Unfreeze all layers
        """
        self.freeze_to(0)
