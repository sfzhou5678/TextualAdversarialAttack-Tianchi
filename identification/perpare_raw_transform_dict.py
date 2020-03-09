from tqdm import tqdm
import json
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from collections import defaultdict, OrderedDict

bert_tokenizer = BertTokenizer.from_pretrained('../data/chinese_vocab.txt', do_lower_case=True)
tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)

freq_dict = json.load(open('freq_dict.fasttext.json', encoding='utf-8'))
## 把word lvl转换成bert lvl的 (哪种更好还不确定, 之后可以对比一下)
new_freq_dict = defaultdict(list)
for char in freq_dict:
  tokens = tokenizer(char)
  for token in tokens:
    new_freq_dict[token].extend(freq_dict[char])
print(len(freq_dict))
freq_dict = new_freq_dict
print(len(freq_dict))
infos = {}
for char in freq_dict:
  infos[char] = {
    'scores': freq_dict[char],
    'cnt': len(freq_dict[char]),
    'mean': np.mean(freq_dict[char]),
    'product': len(freq_dict[char]) * np.mean(freq_dict[char])
  }

## 选出最关键的前25%个词汇，然后准备进行变形
target_chars_transform_dict = OrderedDict()
for char, info in sorted(infos.items(), key=lambda d: d[1]['product'], reverse=True)[:int(len(infos) * 0.25)]:
  print(char, info['cnt'], info['mean'], info['product'])
  target_chars_transform_dict[char] = {
    'scores': info['product'],
    'transforms': []
  }

from manipulate import *
from attackers import is_alpha

phonetic_transform = PhoneticTransform()
# 这是一个需要控制N参数的特例
pronunciation_transform = PronunciationTransform('../data/chaizi/中国所有汉字大全 一行一个.txt')
char_swap_transform = CharSwapTransform()
char_drop_transform = CharDropTransform()
radical_transform = RadicalTransform('../data/chaizi/chaizi-jt.txt')
# homonymic_transform = HomonymicTransform()
hxw_transform = HuoXingWenTransform()

phonetic_char_swap_transform = SequentialModel([phonetic_transform, char_swap_transform])
hxw_radical_transform = SequentialModel([hxw_transform, radical_transform])
radical_chardrop_transform = SequentialModel([radical_transform, char_drop_transform])
hxw_radical_chardroptransform = SequentialModel([hxw_transform, radical_transform, char_drop_transform])


def transform(token):
  transforms = set()
  ## 带随机性质的：
  for _ in range(len(token) * 2):
    if is_alpha(token):
      transforms |= set(char_swap_transform([token], 0))
      transforms |= set(char_drop_transform([token], 0))
    transforms |= set(phonetic_char_swap_transform([token], 0))
    transforms |= set(radical_chardrop_transform([token], 0))
    transforms |= set(hxw_radical_chardroptransform([token], 0))

  ## 固定的：
  transforms |= set(radical_transform([token], 0))
  transforms |= set(hxw_transform([token], 0))
  transforms |= set(hxw_radical_transform([token], 0))

  for tsf in pronunciation_transform([token], 0, N=100):  # 因为这个trans会处理word的情况，所以返回的是len(word)的list
    transforms |= set(tsf)
  return transforms


for char in tqdm(target_chars_transform_dict):
  transforms = transform(char)
  transforms = [t for t in transforms if t not in target_chars_transform_dict]  # 过滤掉同为target的词
  target_chars_transform_dict[char]['transforms'] = transforms
json.dump(target_chars_transform_dict, open('bert_tokenizer_top25%.raw_trans_dict.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
