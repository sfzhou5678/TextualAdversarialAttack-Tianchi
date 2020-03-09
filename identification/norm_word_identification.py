from identification import SingleCharIdentification
from model import FastTextInferenceModel
import jieba
from collections import defaultdict
import json
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
  model_path = '../data/materials/mini.ftz'
  inference_model = FastTextInferenceModel(model_path)

  kw_identification = SingleCharIdentification(inference_model)
  bert_tokenizer = BertTokenizer.from_pretrained('../data/chinese_vocab.txt', do_lower_case=True)
  # tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)
  tokenizer = lambda x: list(jieba.cut(x))  # 用于fasttext

  obscenities = set()
  with open('../data/obscenities.txt', encoding='utf-8') as f:
    for line in f:
      content = line.strip()
      obscenities.add(content)
      # if len(obscenities) >= 100:
      #   break
  obscenities = list(obscenities)

  score_dict = defaultdict(list)
  for text in tqdm(obscenities):
    tokens = tokenizer(text)
    key_tokens = kw_identification(tokens, N=None)
    for idx, score in key_tokens:
      # if score <= 0:
      #   break
      token = tokens[idx]
      score_dict[token].append(score)

  token_scores = [(token, np.mean(score_dict[token])) for token in score_dict]
  token_scores = sorted(token_scores, key=lambda d: d[1], reverse=True)
  with  open('token_scores_dict.fasttext.jieba_tokenizer.json', 'w', encoding='utf-8') as wf:
    for token, score in token_scores:
      try:
        wf.write('%s\t%.6f\n' % (token, score))
      except:
        print(token)
