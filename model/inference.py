import fasttext
import numpy as np
import jieba
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
from model.local_fasttext import FasttextConfig, Fasttext
from model.bert import BertClassificationTransform
import os


class InferenceModel():
  def __init__(self):
    pass

  def inference(self, transformed_text_batch):
    """
    :param transformed_text_batch:
    :return: probs
    """
    raise NotImplementedError()


class FastTextInferenceModel(InferenceModel):
  def __init__(self, model_paths):
    super().__init__()
    if isinstance(model_paths, str):
      model_paths = [model_paths]
    print('Models:', model_paths)
    self.models = [fasttext.load_model(model_path) for model_path in model_paths]
    self.is_obscenity_dict = {'__label__0': 0,
                              '__label__1': 1,
                              '__label__正常': 0,
                              '__label__辱骂': 1}

  def inference(self, transformed_text_batch):
    transformed_text_batch = [' '.join(jieba.cut(text)) for text in transformed_text_batch]  # 和测评代码对齐
    probs = np.zeros(len(transformed_text_batch), dtype=np.float32)
    for model in self.models:
      lbls, cur_probs = model.predict(transformed_text_batch)
      cur_probs = [prob[0] if self.is_obscenity_dict[lbl[0]] else 1 - prob[0] for lbl, prob in zip(lbls, cur_probs)]
      cur_probs = np.array(cur_probs)
      probs += cur_probs / len(self.models)

    return probs


class LocalFasttextInferenceModel(InferenceModel):
  def __init__(self, model, transform, device, batch_size=128):
    super().__init__()
    self.model = model
    self.transform = transform
    self.batch_size = batch_size
    self.device = device

    self.model.eval()

  def inference(self, transformed_text_batch):
    samples = [self.transform(content, idx) for idx, content in enumerate(transformed_text_batch)]
    dataloader = DataLoader(samples, batch_size=self.batch_size, collate_fn=self.transform.batchify, shuffle=False)

    probs = np.array([])
    for step, batch in enumerate(dataloader):
      batch = tuple(t.to(self.device) for t in batch)
      uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = batch

      with torch.no_grad():
        logits = self.model(input_ids_batch)
      prob = F.softmax(logits)
      prob = prob.detach().cpu().numpy()[:, 1]
      probs = np.concatenate([probs, prob], 0)

    return probs


class BertInferenceModel(InferenceModel):
  def __init__(self, model, transform, device, batch_size=16):
    super().__init__()
    self.model = model
    self.transform = transform
    self.batch_size = batch_size
    self.device = device

    self.model.eval()

  def inference(self, transformed_text_batch):
    samples = [self.transform(content, idx) for idx, content in enumerate(transformed_text_batch)]
    dataloader = DataLoader(samples, batch_size=self.batch_size, collate_fn=self.transform.batchify, shuffle=False)

    probs = np.array([])
    for step, batch in enumerate(dataloader):
      batch = tuple(t.to(self.device) for t in batch)
      uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = batch

      with torch.no_grad():
        logits = self.model(input_ids_batch, segment_ids_batch, input_masks_batch)
      prob = F.softmax(logits)
      prob = prob.detach().cpu().numpy()[:, 1]
      probs = np.concatenate([probs, prob], 0)

    return probs


def get_bert_inference_model(bert_model, batch_size=16, max_seq_length=100):
  tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_model, "vocab.txt"), do_lower_case=True)
  transform = BertClassificationTransform(tokenizer=tokenizer, is_test=True, max_len=max_seq_length)

  state_save_path = os.path.join(bert_model, 'model.state')
  if os.path.exists(state_save_path):
    state = torch.load(state_save_path, map_location="cpu")
    model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2,
                                                          state_dict=state['model_state'])
  else:
    previous_model_file = os.path.join(bert_model, "pytorch_model.bin")
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=2)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  inference_model = BertInferenceModel(model, transform, device, batch_size=batch_size)

  return inference_model


def get_fasttext_inference_model(ckpt, batch_size=128, max_seq_length=100):
  tokenizer = BertTokenizer.from_pretrained(os.path.join(ckpt, "vocab.txt"), do_lower_case=True)
  transform = BertClassificationTransform(tokenizer=tokenizer, is_test=True, max_len=max_seq_length)
  config = FasttextConfig(len(tokenizer.vocab))

  previous_model_file = os.path.join(ckpt, "pytorch_model.bin")
  model_state_dict = torch.load(previous_model_file, map_location="cpu")
  model = Fasttext(config)
  model.load_state_dict(model_state_dict)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  inference_model = LocalFasttextInferenceModel(model, transform, device, batch_size=batch_size)

  return inference_model


if __name__ == '__main__':
  bert_model = '../ckpt/clf/ernie_weibo'
  bert_inferencer = get_bert_inference_model(bert_model)

  transformed_text_batch = ['配你妈的字，司马玩意儿', '要点b脸', '你妈死了', '网络暴力的请您死个妈助助兴先']
  print(bert_inferencer.inference(transformed_text_batch))
