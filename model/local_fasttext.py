import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class FasttextConfig(object):
  """配置参数"""

  def __init__(self, n_vocab):
    self.model_name = 'FastText'

    self.dropout = 0.5
    self.num_classes = 2
    self.n_vocab = n_vocab
    self.emb_dim = 300
    self.hidden_size = 256
    # self.n_gram_vocab = 250499  # ngram 词表大小


class Fasttext(nn.Module):
  def __init__(self, config):
    super(Fasttext, self).__init__()
    # if config.embedding_pretrained is not None:
    #   self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
    # else:
    self.embedding = nn.Embedding(config.n_vocab, config.emb_dim, padding_idx=0)
    # self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.emb_dim)
    # self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.emb_dim)
    self.dropout = nn.Dropout(config.dropout)
    self.fc1 = nn.Linear(config.emb_dim, config.hidden_size)
    self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    self.num_labels = config.num_classes

  def forward(self, input_ids, labels=None):
    out_word = self.embedding(input_ids)
    # out_bigram = self.embedding_ngram2(x[2])
    # out_trigram = self.embedding_ngram3(x[3])
    # out = torch.cat((out_word, out_bigram, out_trigram), -1)
    out = out_word
    out = out.mean(dim=1)
    if labels is not None:
      p = 0.7
      # p = 0.9 - random.random() * 0.3
      shuffled_out = out[torch.randperm(out.size(0))]
      mixup_out = p * out + (1 - p) * shuffled_out
      out = torch.cat([out, mixup_out], 0)
      labels = torch.cat([labels, labels])

    out = self.dropout(out)
    out = self.fc1(out)
    out = F.relu(out)
    logits = self.fc2(out)

    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      return loss
    else:
      return logits
