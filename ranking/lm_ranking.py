import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pytorch_pretrained_bert import BertModel, BertConfig, BertForSequenceClassification, BertTokenizer
from model.bert import BertLmRankingTransform
from pytorch_pretrained_bert import BertForPreTraining


class LmBasedRanker():
  def __init__(self, model, tokenizer, transform, device):
    self.model = model
    self.tokenizer = tokenizer
    self.transform = transform
    self.device = device

  def __call__(self, tokens, idx, return_max=False):
    samples = [self.transform(tokens, idx)]
    dataloader = DataLoader(samples, batch_size=1, collate_fn=self.transform.batchify, shuffle=False)

    for batch in dataloader:
      batch = tuple(t.to(self.device) for t in batch)
      input_ids_batch, segment_ids_batch, input_masks_batch = batch

      with torch.no_grad():
        slot_logits, _ = self.model(input_ids_batch, segment_ids_batch, input_masks_batch)
      char_prob = F.softmax(slot_logits[0, idx + 1]).detach().cpu().numpy()  # idx+1 是因为第一个变成了[CLS]
    if return_max:
      return self.tokenizer.convert_ids_to_tokens([char_prob.argmax()])[0]
    else:
      ids_to_tokens = self.tokenizer.ids_to_tokens
      relevance_dict = {}
      for char_id, prob in zip(list(ids_to_tokens.keys()), char_prob):
        char = ids_to_tokens[char_id]
        relevance_dict[char] = prob
      return relevance_dict


def get_lm_ranker(bert_model, max_seq_length=100):
  tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_model, "vocab.txt"), do_lower_case=True)
  transform = BertLmRankingTransform(tokenizer=tokenizer, max_len=max_seq_length)

  state_save_path = os.path.join(bert_model, 'model.state')
  if os.path.exists(state_save_path):
    state = torch.load(state_save_path, map_location="cpu")
    model = BertForPreTraining.from_pretrained(bert_model, state_dict=state['model_state'])
  else:
    previous_model_file = os.path.join(bert_model, "pytorch_model.bin")
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    model = BertForPreTraining.from_pretrained(bert_model, state_dict=model_state_dict)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  lm_ranker = LmBasedRanker(model, tokenizer, transform, device)
  return lm_ranker


if __name__ == '__main__':
  bert_model = '../ckpt/raw/bert-base-chinese'
  max_seq_length = 100

  lm_ranker = get_lm_ranker(bert_model, max_seq_length)

  for text in ['配你妈的字，司马玩意儿', '要点b脸', '你妈死了', '网络暴力的请您死个妈助助兴先']:
    tokens = list(text)
    for idx in range(len(tokens)):
      print(idx, tokens[idx], '->', lm_ranker(tokens, idx, return_max=True))
    print('-' * 80)
