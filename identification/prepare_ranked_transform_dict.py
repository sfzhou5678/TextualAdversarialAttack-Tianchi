import json
from model import FastTextInferenceModel
import jieba
from pytorch_pretrained_bert import BertTokenizer
from collections import defaultdict, OrderedDict
from materials.preprocessing_module import preprocess_text
from performance_evaluator import PerformanceEvaluator
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from attackers import is_alpha

if __name__ == '__main__':
  model_path = '../data/materials/mini.ftz'
  inference_model = FastTextInferenceModel(model_path)

  bert_tokenizer = BertTokenizer.from_pretrained('../data/chinese_vocab.txt', do_lower_case=True)
  tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)
  # tokenizer = lambda x: list(jieba.cut(x))  # 用于fasttext

  obscenities = set()
  with open('../data/obscenities.txt', encoding='utf-8') as f:
    for line in f:
      content = line.strip()
      content = preprocess_text(content)
      obscenities.add(content)
  obscenities = list(obscenities)

  vec_emb_path = '../data/materials/zh.300.vec.gz'
  fasttext_model_path = '../data/materials/mini.ftz'
  fasttext_model = FastTextInferenceModel(fasttext_model_path)
  performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=fasttext_model)  # 模拟远程防御模型，找到强力攻击样本

  target_chars_transform_dict = json.load(open('bert_tokenizer_top25%.raw_trans_dict.json', encoding='utf-8'))
  ranked_transform_dict = OrderedDict()
  for t in target_chars_transform_dict:
    ranked_transform_dict[t] = {
      'scores': target_chars_transform_dict[t]['scores'],
      'transforms': defaultdict(list)
    }
  cnt = 0
  for raw_text in tqdm(obscenities):
    tokens = tokenizer(raw_text)
    cur_token_set = set(tokens)
    for target_token in target_chars_transform_dict:
      if target_token in cur_token_set:
        tgt_indices = [idx for idx, t in enumerate(tokens) if t == target_token]
        baseline_transformed_text = ''.join(
          [token for idx, token in enumerate(tokens) if idx not in tgt_indices])  # drop当baseline
        transformed_texts = [baseline_transformed_text]
        for tsf_token in target_chars_transform_dict[target_token]['transforms']:
          new_tokens = tokens[:]
          for idx in tgt_indices:
            new_tokens[idx] = tsf_token
          transformed_texts.append(''.join(new_tokens))
        ref_texts = [raw_text] * len(transformed_texts)
        soft_scores, hard_scores = performance_evaluator.calc_final_score(ref_texts, transformed_texts,
                                                                          show_details=False)
        soft_scores = [s - soft_scores[0] for s in soft_scores]
        for tsf_token, score in zip(target_chars_transform_dict[target_token]['transforms'], soft_scores[1:]):
          ranked_transform_dict[target_token]['transforms'][tsf_token].append(score)
    cnt += 1
    # if cnt >= 50:
    #   break
  with open('ranked_transform_dict_bert_tokenizer.txt', 'w', encoding='utf-8') as wf:
    for token in ranked_transform_dict:
      if len(ranked_transform_dict[token]['transforms']) == 0:
        continue
      tsf_candidates_scores = [(tsf_token, np.mean(ranked_transform_dict[token]['transforms'][tsf_token])) for tsf_token
                               in
                               ranked_transform_dict[token]['transforms']]
      ranked_candidates = sorted(tsf_candidates_scores, key=lambda d: d[1], reverse=True)
      tsf_tokens = []
      tsf_probs = []

      for t, s in ranked_candidates:
        if t != '' and (len(t) <= 4 or is_alpha(t)):
          tsf_tokens.append(t)
          tsf_probs.append(s * 100)
      if len(tsf_probs) == 0:
        continue
      tsf_probs = list(softmax(tsf_probs))
      ranked_transform_dict[token].pop('transforms')
      ranked_transform_dict[token]['transform_tokens'] = tsf_tokens
      ranked_transform_dict[token]['transform_probs'] = tsf_probs

      wf.write('%s\t%s\n' % (token, json.dumps(ranked_transform_dict[token], ensure_ascii=False)))
