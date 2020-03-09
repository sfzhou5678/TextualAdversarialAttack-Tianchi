import os
import time
import json
import shutil
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

from model.bert import BertClassificationDataset, BertClassificationTransform, warmup_linear
from sklearn.model_selection import GroupShuffleSplit


def eval_running_model(dataloader):
  global eval_loss, step, batch, uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch
  model.eval()
  eval_loss, eval_hit_times = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
  for step, batch in enumerate(dataloader, start=1):
    batch = tuple(t.to(device) for t in batch)
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch = batch

    with torch.no_grad():
      tmp_eval_loss = model(input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch)
      logits = model(input_ids_batch, segment_ids_batch, input_masks_batch)

    logits = logits.detach().cpu().numpy()
    label_ids = labels_batch.to('cpu').numpy()

    eval_hit_times += (logits.argmax(-1) == label_ids).sum()
    eval_loss += tmp_eval_loss.mean().item()

    nb_eval_examples += labels_batch.size(0)
    nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  eval_accuracy = eval_hit_times / nb_eval_examples
  result = {
    'train_loss': tr_loss / nb_tr_steps,
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,

    'epoch': epoch,
    'global_step': global_step,
  }
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Required parameters
  parser.add_argument("--bert_model", default='ckpt/ernie', type=str)
  parser.add_argument("--train_file", default='data/clf/train_adv0.txt', type=str)
  parser.add_argument("--output_dir", default='ckpt/clf/ernie_adv0', type=str)
  parser.add_argument("--max_seq_length", default=100, type=int)
  parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
  parser.add_argument("--eval_batch_size", default=4, type=int, help="Total batch size for eval.")
  parser.add_argument("--print_freq", default=200, type=int, help="Total batch size for eval.")

  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs", default=5.0, type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion", default=0.2, type=float,
                      help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
  parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()
  print(args)
  os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

  ## init dataset and bert model
  tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True)
  transform = BertClassificationTransform(tokenizer=tokenizer, is_test=False, max_len=args.max_seq_length)

  print('=' * 80)
  print('Input file:', args.train_file)
  print('Output dir:', args.output_dir)
  print('=' * 80)

  content_set = set()

  train_samples = []
  with open(args.train_file, encoding='utf-8') as f:
    for line in f:
      sample = line.strip().split('\t')
      if len(sample) >= 3 and sample[0] not in content_set:
        train_samples.append(sample)
        content_set.add(sample[0])
      elif len(sample) >= 2 and not (sample[-1].startswith('obs') or sample[-1].startswith('non')) and sample[
        0] not in content_set:
        sample.append('random_%d' % len(train_samples))
        train_samples.append(sample)
        content_set.add(sample[0])
  random.seed = args.seed
  random.shuffle(train_samples)
  group_ids = [sample[-1] for sample in train_samples]
  train_samples = [sample[:2] for sample in train_samples]
  folds = GroupShuffleSplit(n_splits=5, random_state=args.seed)
  for fold, (trn_idx, val_idx) in enumerate(folds.split(train_samples, groups=group_ids)):
    val_samples = [train_samples[idx] for idx in val_idx]
    train_samples = [train_samples[idx] for idx in trn_idx]
    break
  # train_samples, val_samples = train_samples[:int(len(train_samples) * 0.8)], train_samples[
  #                                                                             int(len(train_samples) * 0.8):]
  train_dataset = BertClassificationDataset(train_samples, transform)
  val_dataset = BertClassificationDataset(val_samples, transform)
  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size, collate_fn=transform.batchify, shuffle=True)
  val_dataloader = DataLoader(val_dataset,
                              batch_size=args.eval_batch_size, collate_fn=transform.batchify, shuffle=False)

  epoch_start = 1
  global_step = 0
  best_eval_loss = float('inf')
  best_test_loss = float('inf')

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
  shutil.copyfile(os.path.join(args.bert_model, 'bert_config.json'), os.path.join(args.output_dir, 'bert_config.json'))

  state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
  previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
  model_state_dict = torch.load(previous_model_file, map_location="cpu")
  model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=2)
  model = model.cuda()

  param_optimizer = list(model.named_parameters())
  param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate)

  device = torch.device("cuda")

  tr_total = int(
    train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
  print_freq = args.print_freq
  eval_freq = min(len(train_dataloader) // 2, 1000)
  print('Print freq:', print_freq, "Eval freq:", eval_freq)

  for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    with tqdm(total=len(train_dataloader)) as bar:
      for step, batch in enumerate(train_dataloader, start=1):
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch = batch
        loss = model(input_ids_batch, segment_ids_batch, input_masks_batch, labels_batch)
        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids_batch.size(0)
        nb_tr_steps += 1

        lr_this_step = args.learning_rate * warmup_linear(global_step / tr_total, args.warmup_proportion)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_this_step
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if step % print_freq == 0:
          bar.update(min(print_freq, step))
          time.sleep(0.02)
          print(global_step, tr_loss / nb_tr_steps)

        if global_step % eval_freq == 0:
          val_result = eval_running_model(val_dataloader)
          print('Global Step %d VAL res:\n' % global_step, val_result)

          if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving at]', state_save_path)
            torch.save(model.state_dict(), state_save_path)
