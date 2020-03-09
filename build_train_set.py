import os
import random
import time
from utils import pickle_load, pickle_dump
from attackers import ObscenityAttacker
from model import FastTextInferenceModel, get_bert_inference_model, get_fasttext_inference_model

rounds, topK = 5, 5  # rounds可以由简至难
thres = 0.8
neg_coef = 4
max_line = 667
output_path = 'data/outputs/output_by_adv0.txt'

# fasttext_model_folder = 'ckpt/clf/fasttext_simple_adv0'
# local_fasttext = get_fasttext_inference_model(fasttext_model_folder, 128, 100)
# kw_identify_model = local_fasttext
# attack_model = local_fasttext

fasttext_model_path = 'data/materials/mini.ftz'
fasttext_model = FastTextInferenceModel(fasttext_model_path)
kw_identify_model = fasttext_model
attack_model = fasttext_model

# bert_model_folder = 'ckpt/clf/ernie_%s' % model_tag
# bert_model = get_bert_inference_model(bert_model_folder, 128, 100)
# kw_identify_model = bert_model
# attack_model = bert_model

obscenities = []
content_set = set()
with open('data/obscenities.txt', encoding='utf-8') as f:
  for line in f:
    content = line.strip()
    if content not in content_set:
      obscenities.append((content, 1, 'obs_%d' % len(obscenities)))
      content_set.add(content)
    if max_line is not None and len(obscenities) >= max_line:
      break
target_cnt = len(obscenities) * neg_coef
print(len(obscenities))

white_list_path = 'data/obscenities_white_list.txt'
white_list_set = set()
with open(white_list_path, encoding='utf-8') as f:
  for line in f:
    white_list_set.add(line.strip())
    if len(white_list_set) >= 0.75 * target_cnt:
      break

tokenizer = lambda x: list(map(str, list(x)))
obs_attacker = ObscenityAttacker(kw_identify_model, attack_model, tokenizer)
new_obscenities, new_obscenities_group_ids = obs_attacker.generate_taa_samples(
  [o[0] for o in obscenities], group_ids=[o[2] for o in obscenities], rounds=rounds, topK=topK)
new_obscenities = [(content, 1, group_id) for content, group_id in zip(new_obscenities, new_obscenities_group_ids)]
obscenities.extend(new_obscenities)
print('After add TAA:', len(obscenities))

non_obscenities = []
for content in white_list_set:
  non_obscenities.append((content, 0, 'non_%d' % len(non_obscenities)))
content_set |= white_list_set

if os.path.exists(output_path):
  with open(output_path, encoding='utf-8') as f:
    lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
      content, score = line.split('\t')
      if content in white_list_set:
        continue
      score = float(score)
      if score >= thres and content not in content_set:
        non_obscenities.append((content, 0, 'non_%d' % len(non_obscenities)))
        content_set.add(content)
      if len(non_obscenities) >= 0.75 * target_cnt:
        break
  print('After add hard samples:', len(non_obscenities))

with open('data/corpus.txt', encoding='utf-8') as f:
  for line in f:
    if len(non_obscenities) >= target_cnt:
      break
    content = line.strip()
    if random.random() < 0.2 and content not in content_set:
      non_obscenities.append((content, 0, 'non_%d' % len(non_obscenities)))
      content_set.add(content)

new_non_obscenities, new_non_obscenities_group_ids = obs_attacker.generate_taa_samples(
  [o[0] for o in non_obscenities], group_ids=[o[2] for o in non_obscenities], rounds=rounds, topK=topK)
new_non_obscenities = [(content, 0, group_id) for content, group_id in
                       zip(new_non_obscenities, new_non_obscenities_group_ids)]
non_obscenities.extend(new_non_obscenities)
obscenities.extend(non_obscenities)

# random.shuffle(obscenities) # shuffle留给train
with open('data/clf/train.txt', 'w', encoding='utf-8') as wf:
  for content, lbl, group_id in obscenities:
    wf.write('%s\t%d\t%s\n' % (content, lbl, group_id))
