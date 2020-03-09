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
from pytorch_pretrained_bert.optimization import BertAdam
from model.bert import BertClassificationDataset, BertClassificationTransform

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Required parameters
  parser.add_argument("--bert_model", default='ckpt/clf/ernie_adv1', type=str)
  parser.add_argument("--file", default='data/corpus.txt', type=str)
  # parser.add_argument("--file", default='data/obscenities.txt', type=str)
  parser.add_argument("--max_seq_length", default=64, type=int)
  parser.add_argument("--eval_batch_size", default=128, type=int, help="Total batch size for eval.")

  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()
  print(args)
  os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

  ## init dataset and bert model
  tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True)
  transform = BertClassificationTransform(tokenizer=tokenizer, is_test=True, max_len=args.max_seq_length)

  print('=' * 80)
  print('Input file:', args.file)
  print('Used model:', args.bert_model)
  print('=' * 80)

  raw_samples = []
  with open(args.file, encoding='utf-8') as f:
    for line in f:
      raw_samples.append(line.strip())
      if len(raw_samples) >= 1000000:
        break
  samples = [transform(content, idx) for idx, content in tqdm(enumerate(raw_samples), total=len(raw_samples))]

  val_dataloader = DataLoader(samples, batch_size=args.eval_batch_size, collate_fn=transform.batchify, shuffle=False)

  state_save_path = os.path.join(args.bert_model, 'model.state')
  if os.path.exists(state_save_path):
    state = torch.load(state_save_path, map_location="cpu")
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2,
                                                          state_dict=state['model_state'])
  else:
    previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=2)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  wf = open('data/outputs/output.txt', 'w', encoding='utf-8')
  for step, batch in tqdm(enumerate(val_dataloader, start=1), total=len(val_dataloader)):
    batch = tuple(t.to(device) for t in batch)
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = batch

    with torch.no_grad():
      logits = model(input_ids_batch, segment_ids_batch, input_masks_batch)
    prob = F.softmax(logits)
    scores = prob.detach().cpu().numpy()[:, 1]
    for idx, score in zip(uuid_batch.detach().cpu().numpy(), scores):
      content = raw_samples[idx]
      wf.write('%s\t%.6f\n' % (content, score))
  wf.close()
