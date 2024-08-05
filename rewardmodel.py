from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
    pipeline
)
from transformers.utils import PaddingStrategy
from tqdm.notebook import trange
from typing import Any, Dict, List, Optional, Union
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from trl.core import LengthSampler

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k)
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss.mean()


class RewardModel:
    def __init__(self, train_dataset, val_dataset, max_len=512, resume_from_checkpoint=False,
                 output_name='output', learning_rate=1e-4,
                 per_device_train_batch_size=4, per_device_eval_batch_size=1,
                 num_train_epochs=1, weight_decay=1e-3, gradient_accumulation_steps=1,
                 gradient_checkpointing=False, bf16=True, deepspeed=None, local_rank=-1,
                 optim='adamw_hf', lr_scheduler_type='linear', seed=42):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_len = max_len
        self.resume_from_checkpoint = resume_from_checkpoint

        self.output_name = output_name
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.bf16 = bf16
        self.deepspeed = deepspeed
        self.local_rank = local_rank
        self.optim = optim
        self.lr_scheduler_type = lr_scheduler_type
        self.seed = seed
        self.accuracy = evaluate.load("accuracy")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=1,
            torch_dtype=torch.float16
        ).float()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", use_auth_token=True)

        self._setup_training()

    def _preprocess_function(self, dataset):
        # j is the preferred sentence
        new_dataset = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for pos_example, neg_example in zip(dataset["pos"], dataset["neg"]):
            tokenized_j = self.tokenizer(pos_example, truncation=True)
            tokenized_k = self.tokenizer(neg_example, truncation=True)

            new_dataset["input_ids_j"].append(tokenized_j["input_ids"])
            new_dataset["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_dataset["input_ids_k"].append(tokenized_k["input_ids"])
            new_dataset["attention_mask_k"].append(tokenized_k["attention_mask"])

        return new_dataset

    def _setup_training(self):

        # 1. Prepare peft_config
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]
        )

        # 2. Initialize model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = not self.gradient_checkpointing
        num_proc = None
        original_columns = self.train_dataset.column_names

        # 3. Upload datasets
        self.train_dataset = self.train_dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        self.train_dataset = self.train_dataset.filter(
            lambda x: len(x["input_ids_j"]) <= self.max_len and len(x["input_ids_k"]) <= self.max_len
        )

        self.val_dataset = self.val_dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        self.val_dataset = self.val_dataset.filter(
            lambda x: len(x["input_ids_j"]) <= self.max_len and len(x["input_ids_k"]) <= self.max_len
        )

        # 4. State training arguments (we pass them to Trainer)
        training_args = TrainingArguments(
            output_dir=self.output_name,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            save_strategy="epoch",
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            deepspeed=self.deepspeed,
            local_rank=self.local_rank,
            remove_unused_columns=False,
            label_names=[],
            logging_strategy="epoch",
            optim=self.optim,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
        )

        # 5. Training model
        self.trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=RewardDataCollatorWithPadding(tokenizer=self.tokenizer,
                                                        padding='max_length', max_length=self.max_len),
        )


    def fit(self):
        self.trainer.train(self.resume_from_checkpoint)
        print("Saving last checkpoint of the model")
        self.model.save_pretrained(self.output_name + "_peft_last_checkpoint")