"""
同源操作，即通过drop、swap等操作实现数据变形
"""
import re
import numpy as np
import random
from manipulate.transform import Transform


class TokenSwapTransform(Transform):
  def __init__(self, range=(1, 2)):
    """
    交换tokens[idx]和tokens[idx+offset]
    """
    super().__init__()
    self.range = range

  def __call__(self, tokens, idx):
    if len(tokens) <= 4:
      return tokens[:]

    range = self.range
    offset = np.random.randint(range[0], range[1] + 1, size=1)[0]
    if np.random.random() <= 0.5:
      offset *= -1
    while idx + offset < 0 or idx + offset >= len(tokens):
      offset = np.random.randint(range[0], range[1] + 1, size=1)[0]
      if np.random.random() <= 0.5:
        offset *= -1

    new_tokens = tokens[:]
    new_tokens[idx], new_tokens[idx + offset] = new_tokens[idx + offset], new_tokens[idx]

    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class CharSwapTransform(Transform):
  def __init__(self):
    """
    交换tokens[idx]内部的char的位置
    """
    super().__init__()

  def __call__(self, tokens, idx):
    """
    直接写成了单词乱序
    :param tokens:
    :param idx:
    :return:
    """
    target_token = tokens[idx]
    new_token = self._transform(target_token)
    if target_token == new_token:
      return None

    new_tokens = tokens[:]
    new_tokens[idx] = new_token
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, target_token):
    if len(target_token) < 4:
      return target_token

    middle_chars = list(target_token[1:-1])
    if len(set(middle_chars)) > 1:
      while middle_chars == list(target_token[1:-1]):
        random.shuffle(middle_chars)
    new_token = target_token[0] + ''.join(middle_chars) + target_token[-1]
    return new_token


class AddTransform(Transform):
  def __init__(self, pool=None):
    super().__init__()
    if pool is not None:
      self.pool = list(pool)
    else:
      self.pool = list('儍和如最助酒喜宋很好谢爱啵俺伱妮鉨蔴玛么嚒麽俵肝欢喜鹿美国咕休息朋友乐仓月皇帝达拉然芙安静' +
                       '狖虺瓞呶薁皴鸨媾贲' +
                       '1356790adehijkloqrstuvxyz' +
                       '，。！,.:…-/!()[]"_|=;')

  def __call__(self, tokens, idx):
    # cur_char_pool = list(set(''.join(tokens)))
    noise = random.choice(self.pool)
    new_tokens = tokens[:idx] + [noise] + tokens[idx:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    noise = random.choice(self.pool)  # 用同一个噪声，克服jaccard指标上的劣势
    for idx in indices:  # 默认已经是降序排列
      new_tokens.insert(idx, noise)
    return new_tokens


class SepInsertTransform(Transform):
  def __init__(self):
    super().__init__()
    self.pool = list(' _|`~-')

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = self._transform(target_token)
    if new_token == target_token:
      return None
    new_tokens = tokens[:idx] + [new_token] + tokens[idx:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, target_token):
    if len(target_token) < 2:
      return target_token
    insert_idx = random.randint(1, len(target_token) - 1)  # 不插头尾
    new_token = target_token[:insert_idx] + '-' + target_token[insert_idx:]
    return new_token


class RandomReplaceTransform(Transform):
  def __init__(self):
    super().__init__()
    self.pool = list('儍和如最助酒喜宋很好谢爱啵俺伱妮鉨蔴玛么嚒麽俵肝欢喜鹿美国咕休息朋友乐仓月皇帝达拉然芙安静' +
                     '1356790adehijkloqrstuvxyz' +
                     '，。！,.:…-/!()[]"_=;`~')

  def __call__(self, tokens, idx):
    cur_char_pool = list(set(''.join(tokens)))
    if not cur_char_pool:
      return None
    noise = random.choice(cur_char_pool + self.pool)

    new_tokens = tokens[:]
    new_tokens[idx] = noise
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


# class EntireAddTransform(Transform):
#   def __init__(self):
#     super().__init__()
#     self.insert_pool = ['(', ')', '!',
#                         'oh', 'y', 'o', 'td', ]
#
#   def __call__(self, tokens, N=5):
#     indices = np.random.choice(list(range(len(tokens))), min(N, len(tokens)), replace=False)
#     indices = sorted(indices, reverse=True)
#     noise = random.choice(self.insert_pool)
#     new_tokens = tokens[:]
#     for idx in indices:
#       new_tokens.insert(idx, noise)
#
#     if self.debug:
#       self.transformed_tokens.append(new_tokens)
#     return new_tokens


class BracketsTransform(Transform):
  def __init__(self):
    super().__init__()
    self.left_bracket = '('
    self.right_bracket = ')'

  def __call__(self, tokens, idx):
    new_tokens = tokens[:idx] + [self.left_bracket + tokens[idx] + self.right_bracket] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class TokenDropTransform(Transform):
  def __init__(self):
    super().__init__()

  def __call__(self, tokens, idx):
    """
    drop的容易导致两种bad case：
    1) 丢掉了关键词，导致句子不像垃圾话
    2) 丢掉了无关紧要的词，导致很容易被识别成垃圾话
    :param tokens:
    :param idx:
    :return:
    """
    new_tokens = tokens[:idx] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class CharDropTransform(Transform):
  def __init__(self):
    super().__init__()

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    if len(target_token) > 1:
      random_idx = random.randint(0, len(target_token) - 1)
      new_token = target_token[:random_idx] + target_token[random_idx + 1:]

      new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    else:
      new_tokens = tokens[:idx] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class SegmentShuffleTransform(Transform):
  def __init__(self, window_size=2):
    super().__init__()
    self.pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'

    self.window_size = window_size

  def __call__(self, tokens, idx):
    swap_offsets = np.random.randint(0, self.window_size + 1, size=len(tokens))
    swap_offsets[0] = 0
    swap_offsets[-1] = 0

    new_tokens = tokens[:]
    for i, offset in enumerate(swap_offsets):
      if offset:
        i_offset = min(len(new_tokens) - 2, i + offset)
        new_tokens[i], new_tokens[i_offset] = new_tokens[i_offset], new_tokens[i]
        swap_offsets[i_offset] = 0
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def segment_transform(self, text, window_size=2):
    """
    整句替换
    替换的基本原则: 根据标点符号分句之后，对于每个句子，要求保证首尾不变，中间的最多和临近的window_size个词交换
    :param text_batch:
    :param window_size:
    :return:
    """
    sub_sentences = re.split(self.pattern, text)
    punctuations = re.findall(self.pattern, text)
    assert len(punctuations) == len(sub_sentences) - 1

    new_sub_sentences = []
    for sub_sentence in sub_sentences:
      if len(sub_sentence) <= 2:
        new_sub_sentences.append(sub_sentence)
        continue
      swap_offsets = np.random.randint(0, window_size + 1, size=len(sub_sentence))
      swap_offsets[0] = 0
      swap_offsets[-1] = 0

      new_sub_sentence = list(sub_sentence)
      for i, offset in enumerate(swap_offsets):
        if offset:
          i_offset = min(len(new_sub_sentence) - 2, i + offset)
          new_sub_sentence[i], new_sub_sentence[i_offset] = new_sub_sentence[i_offset], new_sub_sentence[i]
          swap_offsets[i_offset] = 0
      new_sub_sentences.append(''.join(new_sub_sentence))

    new_text = ''
    for sub_sentence, punc in zip(new_sub_sentences, punctuations):
      new_text += sub_sentence + punc
    new_text += new_sub_sentences[-1]

    return new_text


if __name__ == '__main__':
  ## 整句变型
  t = ['伱zhegou只hehg看着情说出苨家de悲惨身世敢讲话只能在家默默吃shi']
  transform = SegmentShuffleTransform()
  for text in t:
    print(''.join(transform(list(text), 0)))
  # char_swop_transform = CharSwapTransform()
  # print(char_swop_transform)
