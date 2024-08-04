from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.notebook import trange

from trl.core import LengthSampler

class PrepareData:
  def __init__(self, size=20000):
    self.splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
    self.train_binary_df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + self.splits["train"])
    self.val_binary_df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + self.splits["test"])

    self.size = size

    self.train_pairwise_df = self._make_pairwise(self.train_binary_df)
    self.val_pairwise_df = self._make_pairwise(self.val_binary_df)

    self.train_pairwise = Dataset.from_pandas(self.train_pairwise_df)
    self.val_pairwise = Dataset.from_pandas(self.val_pairwise_df)

  def _make_pairwise(self, df):
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    df_pairwise = pd.DataFrame(columns=['pos', 'neg'])
    for pair_index in trange(self.size):
        pos_index = np.random.choice(len(df_pos))
        neg_index = np.random.choice(len(df_neg))
        pos_row = df_pos.iloc[pos_index]
        neg_row = df_neg.iloc[neg_index]
        df_pairwise.loc[len(df_pairwise)] = {'pos': pos_row.text, 'neg': neg_row.text}
    return df_pairwise

class PreparePrompts:
    def __init__(self, tokenizer, min_len=5, max_len=20):
        self.train_dataset = load_dataset("stanfordnlp/imdb", split="train")
        self.val_dataset = load_dataset("stanfordnlp/imdb", split="test")

        self.tokenizer = tokenizer
        self.input_size = LengthSampler(min_len, max_len)

        self.train_dataset = self.train_dataset.map(
            self._crop_prompt,
        )
        self.train_dataset.set_format(type="torch")
        self.val_dataset = self.val_dataset.map(
            self._crop_prompt,
        )
        self.val_dataset.set_format(type="torch")

    def _crop_prompt(self, sample):
        sample["input_ids"] = self.tokenizer.encode(sample["text"])[:self.input_size()]
        sample["query"] = self.tokenizer.decode(sample["input_ids"])
        return sample