# 加载Python自带 或通过pip安装的模块
import json
import os
from functools import partial
import multiprocessing as mp
# import pathos.multiprocessing as mp
import numpy as np
import random
import time


def _do_attack(inp_lines, kw_freq_dict):
  import json
  import os
  from model import get_bert_inference_model, FastTextInferenceModel, get_fasttext_inference_model
  from attackers import ObscenityAttacker, RuleBasedAttacker
  from pytorch_pretrained_bert import BertTokenizer
  import jieba
  fasttext_model_path = 'data/materials/mini.ftz'
  # fasttext_model_path = ['data/materials/mini.ftz', 'data/materials/mini-explicit-labels.ftz']
  fasttext_model = FastTextInferenceModel(fasttext_model_path)

  # bert_model_folder = 'ckpt/clf/ernie_weibo'
  # bert_model = get_bert_inference_model(bert_model_folder, 128, 100)

  kw_identify_model = fasttext_model
  attack_model = fasttext_model

  bert_tokenizer = BertTokenizer.from_pretrained('data/chinese_vocab.txt', do_lower_case=True)
  tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)
  # tokenizer = lambda x: list(jieba.cut(x))

  obs_attacker = ObscenityAttacker(kw_identify_model, attack_model, tokenizer)
  obs_attacker.kw_freq_dict = kw_freq_dict
  out_lines, local_scores = obs_attacker.attack(inp_lines, rounds=31, topK=10)
  return out_lines, local_scores


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  if os.path.exists('/tcdata/benchmark_texts.txt'):
    inp_path = '/tcdata/benchmark_texts.txt'
    max_line = None
  else:
    inp_path = 'data/obscenities.txt'
    max_line = 100

  out_path = 'adversarial.txt'

  inp_lines = []
  with open(inp_path, 'r', encoding='utf-8') as f:
    for line in f:
      inp_lines.append(line.strip())
      if max_line is not None and len(inp_lines) >= max_line:
        break

  time0 = time.time()
  kw_freq_dict = mp.Manager().dict()
  n_cpu = mp.cpu_count()
  max_cpu = 4
  if max_cpu is not None:
    n_cpu = min(n_cpu, max_cpu)
  if n_cpu > 1:
    inp_lines = list(enumerate(inp_lines))
    random.shuffle(inp_lines)
    indices_map = {raw_idx: cur_idx for cur_idx, (raw_idx, line) in enumerate(inp_lines)}
    inp_lines = [line for idx, line in inp_lines]
    with mp.Pool(processes=n_cpu) as p:
      samples_split = np.array_split(inp_lines, n_cpu)
      pool_results = p.map(partial(_do_attack, kw_freq_dict=kw_freq_dict), samples_split)

    out_lines = list(np.concatenate([results[0] for results in pool_results]))
    out_lines = [out_lines[indices_map[idx]] for idx in range(len(out_lines))]
    local_scores = list(np.concatenate([results[1] for results in pool_results]))
  else:
    out_lines, local_scores = _do_attack(inp_lines, kw_freq_dict)
  print(sum(local_scores) / len(local_scores))
  print('Time:', time.time() - time0)
  try:
    target = json.dumps({'text': out_lines}, ensure_ascii=False)
    with open(out_path, 'w', encoding='utf-8') as f:
      f.write(target)
  except:
    from materials.preprocessing_module import preprocess_text

    out_lines = [preprocess_text(line) for line in out_lines]
    target = json.dumps({'text': out_lines}, ensure_ascii=False)
    with open(out_path, 'w', encoding='utf-8') as f:
      f.write(target)
