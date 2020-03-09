import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pytorch_pretrained_bert import BertModel, BertConfig, BertForSequenceClassification, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam


def warmup_linear(x, warmup=0.002):
  if x < warmup:
    return x / warmup
  return 1.0 - x


class BertClassificationTransform(object):
  def __init__(self, tokenizer, is_test, max_len=128):
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.is_test = is_test

  def __call__(self, text, index, label=None):
    uuid = index

    tokens = self.tokenizer.tokenize(text)
    tokens = tokens[:self.max_len - 2]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_masks = [1] * len(input_ids)
    padding = [0] * (self.max_len - len(input_ids))
    input_ids += padding
    input_masks += padding
    segment_ids += padding

    assert len(input_ids) == self.max_len
    assert len(input_masks) == self.max_len
    assert len(segment_ids) == self.max_len

    if self.is_test:
      return uuid, input_ids, segment_ids, input_masks
    else:
      return uuid, input_ids, segment_ids, input_masks, label

  def batchify(self, batch):
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = [], [], [], []
    if not self.is_test:
      labels_batch = []
    for sample in batch:
      uuid, input_ids, segment_ids, input_masks = sample[:4]

      uuid_batch.append(uuid)
      input_ids_batch.append(input_ids)
      segment_ids_batch.append(segment_ids)
      input_masks_batch.append(input_masks)
      if not self.is_test:
        labels_batch.append(sample[-1])

    long_tensors = [uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch]
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = (
      torch.tensor(t, dtype=torch.long) for t in long_tensors)
    if self.is_test:
      return uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch
    else:
      labels_batch = torch.tensor(labels_batch, dtype=torch.long)
      return uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch


class BertLmRankingTransform(object):
  def __init__(self, tokenizer, max_len=128):
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __call__(self, tokens, idx):
    tokens = tokens[:]
    tokens[idx] = '[MASK]'
    tokens = tokens[:self.max_len - 2]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_masks = [1] * len(input_ids)
    padding = [0] * (self.max_len - len(input_ids))
    input_ids += padding
    input_masks += padding
    segment_ids += padding

    assert len(input_ids) == self.max_len
    assert len(input_masks) == self.max_len
    assert len(segment_ids) == self.max_len

    return input_ids, segment_ids, input_masks

  def batchify(self, batch):
    input_ids_batch, segment_ids_batch, input_masks_batch = [], [], []
    for sample in batch:
      input_ids, segment_ids, input_masks = sample
      input_ids_batch.append(input_ids)
      segment_ids_batch.append(segment_ids)
      input_masks_batch.append(input_masks)

    long_tensors = [input_ids_batch, segment_ids_batch, input_masks_batch]
    input_ids_batch, segment_ids_batch, input_masks_batch = (
      torch.tensor(t, dtype=torch.long) for t in long_tensors)

    return input_ids_batch, segment_ids_batch, input_masks_batch


class BertClassificationDataset(Dataset):
  def __init__(self, samples, transform):
    self.data_source = samples
    self.transformed_data = {}
    self.transform = transform

  def __len__(self):
    return len(self.data_source)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      text = self.data_source[index][0]
      label = int(self.data_source[index][1])
      key_data = self.transform(text, index, label)
      self.transformed_data[index] = key_data

      return key_data
