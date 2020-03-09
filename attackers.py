"""
完整的变换的工作流：
- 拼音变换
- shape变换(特别是针对英文、数字)
- 可以在最后上一个关键词识别，然后drop掉一些词
"""
from collections import defaultdict
from performance_evaluator import PerformanceEvaluator
from model import FastTextInferenceModel, BertInferenceModel, get_bert_inference_model
from identification import *
from manipulate import *
from tqdm import tqdm
import re
import numpy as np
import random
from utils import is_alpha
from materials.preprocessing_module import preprocess_text


class ObscenityAttacker():
  def __init__(self, kw_identify_model, attack_model, tokenizer):
    self.kw_freq_dict = {}
    self.local_kw_freq_dict = {}  # 全局freq_dict的近似拷贝，用在需要反复请求的地方，避免全局dict进程同步时的损耗

    self.kw_identification = SingleCharIdentification(kw_identify_model)
    self.tokenizer = tokenizer

    vec_emb_path = 'data/materials/zh.300.vec.gz'
    self.performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=attack_model)  # 模拟远程防御模型，找到强力攻击样本

    self.init_transforms()

  def init_transforms(self):
    random_replace_transform = RandomReplaceTransform()
    token_swap_transform = TokenSwapTransform()
    char_swap_transform = CharSwapTransform()
    add_transform = AddTransform()
    add_sep_transform = AddTransform('_ |')  # 专门添加分隔符的add
    token_drop_transform = TokenDropTransform()
    char_drop_transform = CharDropTransform()
    char_shape_transform = ShapeTransform()

    phonetic_transform = PhoneticTransform()
    case_transform = CaseTransform(first_letter_only=True)
    phonetic_firstletter_transform = PhoneticTransform(first_letter=True)
    radical_transform = RadicalTransform('data/chaizi/chaizi-jt.txt', max_radicals_lengths=2)
    pronunciation_transform = PronunciationTransform('data/chaizi/中国所有汉字大全 一行一个.txt', N=50)

    same_radical_transform = SimpleSameRadicalTransform('data/chaizi/chaizi-jt.txt', max_radicals_lengths=2)

    hxw_transform = HuoXingWenTransform()
    emb_euclidean_transform = EmbeddingTransform('data/index/annoy.euclidean.10neighbors.txt')
    emb_cosine_transform = EmbeddingTransform('data/index/annoy.cosine.10neighbors.txt')

    phonetic_char_swap_transform = SequentialModel([phonetic_transform, char_swap_transform])
    phonetic_char_drop_transform = SequentialModel([phonetic_transform, char_drop_transform])
    phonetic_add_sep_transform = SequentialModel([phonetic_transform, add_sep_transform])
    phonetic_char_shape_transform = SequentialModel([phonetic_transform, char_shape_transform])

    hxw_radical_transform = SequentialModel([hxw_transform, radical_transform])
    radical_chardrop_transform = SequentialModel([radical_transform, char_drop_transform])
    hxw_radical_chardroptransform = SequentialModel(
      [hxw_transform, radical_transform, char_drop_transform])

    homonymic_transform = HomonymicTransform()
    # homonymic_char_shape_transform = HomonymicTransform([char_shape_transform])  # fixme: 临时做法
    # global_transform = GlobalTransform([
    #   # phonetic_char_swap_transform,
    #   # phonetic_add_sep_transform,
    #   phonetic_char_shape_transform,
    #   same_radical_transform,
    #   # pronunciation_transform,
    #   # phonetic_firstletter_transform,
    #   # hxw_transform,
    # ])
    ## ------------------------------------------------------------------------------------------------
    self.global_transforms = [
      homonymic_transform,
      # homonymic_char_shape_transform,
      # global_transform,  # 可读性还是太差，无法实用
    ]

    ## 英文字母相关的操作
    self.alpha_transforms = [
      # char_drop_transform,
      char_swap_transform,
      char_shape_transform,
      add_sep_transform,
      case_transform
    ]

    ## 可能需要执行多轮的操作，都是一些有多个candidates的关键transform
    self.multi_rounds_transforms = [
      # self.emb_cosine_transform,
      emb_euclidean_transform,
      pronunciation_transform,
      same_radical_transform,
      homonymic_transform
    ]

    ## 带一定随机性的或者是有多个candidates但是不关键的transform
    self.random_transforms = [
      add_transform,
      random_replace_transform,

      radical_chardrop_transform,
      hxw_radical_chardroptransform,

      phonetic_char_swap_transform,  # fixme: 这几个也许可以去掉，因为log显示这几个帮助不大，而且其实和alpha_trans有些重复
      # phonetic_char_drop_transform,
      phonetic_add_sep_transform,
      phonetic_char_shape_transform,

      # token_swap_transform  # 比较辣鸡
    ]

    ## 变换模式固定的transforms:
    self.fixed_transforms = [
      token_drop_transform,
      radical_transform,
      # phonetic_transform,
      phonetic_firstletter_transform,
      hxw_transform,
      hxw_radical_transform,
    ]

    ## 会进行多点替换的操作
    self.multi_ptr_transforms = [
      # phonetic_char_swap_transform,
      # phonetic_add_sep_transform,
      # phonetic_char_shape_transform,
      # add_transform
    ]

  def _append_transformed_tokens(self, historical_taa_set, candidate_taas, transformed_tokens):
    if not transformed_tokens:
      return
    preprocessed_transform_text = preprocess_text(''.join(transformed_tokens))
    if preprocessed_transform_text in historical_taa_set:
      return
    if preprocessed_transform_text not in candidate_taas:
      candidate_taas[preprocessed_transform_text] = transformed_tokens
      historical_taa_set.add(preprocessed_transform_text)

  def attack(self, raw_texts, rounds=5, topK=5, debug=False, kw_freq_thres=20.0):
    print('Round:', rounds, 'TopK:', topK)
    local_scores = []
    transformed_texts = []
    for i_text, raw_text in tqdm(enumerate(raw_texts), total=len(raw_texts)):
      best_score = 0.0
      raw_tokens = self.tokenizer(raw_text)
      kw_freqs = []
      for token in raw_tokens:
        if token not in self.kw_freq_dict:
          self.kw_freq_dict[token] = 0
        self.kw_freq_dict[token] += 5
        kw_freqs.append(self.kw_freq_dict[token])
      self.local_kw_freq_dict = self.kw_freq_dict.copy()  # 复制一个全局dict的副本，在高频次query时使用本地副本可以避免进程同步带来的巨大同步耗时
      mean_freq = np.mean(kw_freqs)
      best_transformed_text = raw_text
      best_transformed_tokens = raw_tokens

      ## todo: 可以改成tokens中见过的词太少的话(平均频次低于阈值)，就换成kw idf模式
      # if i_text <= kw_idf_cnt:
      if mean_freq < kw_freq_thres:
        kw_scores = self.kw_identification(raw_tokens, len(raw_tokens))
        kw_scores = [score for _, score in kw_scores]

      preprocessed_raw_text = preprocess_text(''.join(raw_tokens))
      historical_taas = {preprocessed_raw_text}
      candidate_taas = {}
      ##############################################################
      ### Global transform: 整个句子全都替换掉, 然后用这些样本当做种子
      ##############################################################
      # 替换掉所有骂人的关键词
      for transform in self.global_transforms:
        for i in range(topK):  # 调大的话效果会好一点
          self._append_transformed_tokens(historical_taas, candidate_taas, transform.global_transform(raw_tokens))
      if len(candidate_taas) == 0:
        candidate_taas = {preprocessed_raw_text: raw_tokens}

      cur_rounds = rounds  # 当前text的运行轮数，根据长度进行调整
      if len(raw_tokens) < 50:  # 30不会，50不确定
        cur_rounds = int(cur_rounds * (1.5 - 0.1 * len(raw_tokens) // 10))
      for round in range(1, cur_rounds + 1):
        cur_tokens_list = [candidate_taas[text] for text in candidate_taas]
        for tokens_idx, tokens in enumerate(cur_tokens_list):
          if len(tokens) == 0:
            continue
          # # 暴力多点交叉遗传攻击, 肉眼观察较差，但是线上较强
          # for other_tokens_idx, other_tokens in enumerate(cur_tokens_list):
          #   if other_tokens_idx == tokens_idx or len(tokens) != len(other_tokens):
          #     continue
          #
          #   for ratio in [2]:
          #     if len(tokens) < ratio:
          #       continue
          #     new_tokens1 = tokens[:]
          #     new_tokens2 = other_tokens[:]  # 虽然for循环本身就会遍历到(i,j)和(j,i)的情况，但是多来一次可以增加多样性
          #     target_token_indices = np.random.choice(len(other_tokens), len(other_tokens) // ratio, replace=False)
          #     for idx in target_token_indices:
          #       if idx < len(new_tokens1):
          #         new_tokens1[idx] = other_tokens[idx]
          #       if idx > len(new_tokens2):
          #         new_tokens2[idx] = tokens[idx]
          #     self._append_transformed_tokens(historical_taas, candidate_taas, new_tokens1)
          #     self._append_transformed_tokens(historical_taas, candidate_taas, new_tokens2)
          pass

          # ## cross over遗传攻击, 线下&肉眼较强，但是线上很差
          # for other_tokens_idx, other_tokens in enumerate(cur_tokens_list):
          #   if other_tokens_idx == tokens_idx:
          #     continue
          #
          #   try:
          #     tgt_idx = random.randint(3, min(len(tokens), len(other_tokens)) - 3)  # 头尾几个点不截取
          #     new_tokens1 = tokens[:tgt_idx] + other_tokens[tgt_idx:]
          #     new_tokens2 = other_tokens[:tgt_idx] + tokens[tgt_idx:]
          #     self._append_transformed_tokens(historical_taas, candidate_taas, new_tokens1)
          #     self._append_transformed_tokens(historical_taas, candidate_taas, new_tokens2)
          #   except:
          #     pass
          pass
          idx_probs = None
          if round % 2:
            try:
              if mean_freq < kw_freq_thres:
                freqs = kw_scores  # 可能会因为add、drop导致idx错位，不过暂时先忽略
                freqs = freqs[:len(tokens)]
                freqs += [0] * (len(tokens) - len(freqs))
                freqs = np.array(freqs)
                freqs = freqs - freqs.min() + 0.01
              else:
                # fixme: 这里可以改成local_kw来提速如果有必要的话
                freqs = np.array([self.kw_freq_dict[token] if token in self.kw_freq_dict else 1 for token in tokens])
              idx_probs = freqs / freqs.sum()
            except:
              pass
          idx = np.random.choice(list(range(len(tokens))), 1, p=idx_probs)[0]  # 针对关键词的定向攻击
          indices = np.random.choice(list(range(len(tokens))), min(3, len(tokens)), p=idx_probs)  # 批量替换

          ## 开始单点替换
          if is_alpha(tokens[idx]) and len(tokens[idx]) >= 4:
            for transform in self.alpha_transforms:
              self._append_transformed_tokens(historical_taas, candidate_taas, transform(tokens, idx))

          # if len(tokens[idx]) > 1:
          #   ## 对于非英文的、经过转换的token，直接continue掉避免影响可读性。
          #   # (本来可能是拆分成偏旁，然后偏旁->变成别的东西，或者te -> t恶之类的)
          #   # 对速度影响不大，说明这类样本本身并不是很多
          #   continue

          for transform in self.multi_rounds_transforms:
            for _ in range(3):
              self._append_transformed_tokens(historical_taas, candidate_taas, transform(tokens, idx))

          for transform in self.random_transforms:
            self._append_transformed_tokens(historical_taas, candidate_taas, transform(tokens, idx))

          for transform in self.fixed_transforms:
            self._append_transformed_tokens(historical_taas, candidate_taas, transform(tokens, idx))

          ## 开始批量替换，主要是为拼音\add等不会严重影响可读性方法服务，克服这些方法在jaccard指标上的劣势
          indices = sorted(indices, reverse=True)  # 降序排列，为add服务
          for transform in self.multi_ptr_transforms:
            self._append_transformed_tokens(historical_taas, candidate_taas,
                                            transform.multi_ptr_trans(tokens, indices))

        # 挑选出K个攻击力最强的样本，进行下一轮迭代
        cur_transformed_texts = []
        cur_transformed_tokens = []
        for text in candidate_taas:
          cur_transformed_texts.append(text)
          cur_transformed_tokens.append(candidate_taas[text])
        ref_texts = [raw_text] * len(cur_transformed_texts)
        soft_scores, hard_scores = self.performance_evaluator.calc_final_score(ref_texts, cur_transformed_texts,
                                                                               show_details=False)

        ## 词频加权的最终得分，该策略用于对抗线上的自动防御机制
        freqs = np.array(
          [sum([self.local_kw_freq_dict[token] if token in self.local_kw_freq_dict else 1 for token in tokens])
           for tokens in cur_transformed_tokens])
        freq_weights = (freqs - freqs.min()) / (freqs.max() - freqs.min())
        freq_weights = 1.0 - 0.2 * freq_weights
        soft_scores *= freq_weights
        sorted_eval_scores = sorted(enumerate(soft_scores), key=lambda d: d[1], reverse=True)
        if sorted_eval_scores[0][1] > best_score:
          best_score = sorted_eval_scores[0][1]
          best_transformed_text = cur_transformed_texts[sorted_eval_scores[0][0]]
          best_transformed_tokens = cur_transformed_tokens[sorted_eval_scores[0][0]]
          # best_transformed_tokens = self.tokenizer(best_transformed_text) # 额外tokenize一下好像没什么区别，速度也没有影响
          candidate_taas = {}
        else:
          candidate_taas = {best_transformed_text: best_transformed_tokens}
        for idx, score in sorted_eval_scores[:topK]:
          candidate_taas[cur_transformed_texts[idx]] = cur_transformed_tokens[idx]
          # candidate_taas[cur_transformed_texts[idx]] = self.tokenizer(cur_transformed_texts[idx])

        # 然后额外随机选择2个弱鸡模型加到下一轮迭代中去，以保证样本多样性, 线上完全没用
        # try:
        #   extra_cnt = 2
        #   probs = np.array([score for idx, score in sorted_eval_scores[topK:]])  # 从topk以外的样本中选
        #   probs = probs / probs.sum()
        #   rnd_sample_indices = np.random.choice(list(range(topK, len(sorted_eval_scores))), extra_cnt, replace=False,
        #                                         p=probs)
        #   for idx in rnd_sample_indices:
        #     idx = sorted_eval_scores[idx][0]
        #     candidate_taas[cur_transformed_texts[idx]] = cur_transformed_tokens[idx]
        #     # candidate_taas[cur_transformed_texts[idx]] = self.tokenizer(cur_transformed_texts[idx])
        # except:
        #   pass
        pass
      for token in best_transformed_tokens:
        if token not in self.kw_freq_dict:
          self.kw_freq_dict[token] = 0
        self.kw_freq_dict[token] += 2

      transformed_texts.append(best_transformed_text)
      local_scores.append(best_score)

      if debug:
        ## 算贡献度
        for transform in self.transforms:
          tokens_list = transform.transformed_tokens
          if not tokens_list:
            continue
          cur_transformed_texts = list(set([preprocess_text(''.join(tokens)) for tokens in tokens_list]))
          ref_texts = [raw_text] * len(cur_transformed_texts)
          soft_scores, hard_scores = self.performance_evaluator.calc_final_score(ref_texts, cur_transformed_texts,
                                                                                 show_details=False)
          transform.mean_scores.append(np.mean(soft_scores))
          transform.max_scores.append(np.max(soft_scores))
          transform.clear()
    if debug:
      print('-' * 80)
      print('Mean of Mean scores:')
      print('-' * 80)
      score_records = []
      for transform in self.transforms:
        scores = transform.mean_scores
        score = 0
        if scores:
          score = np.mean(scores)
        score_records.append((transform, score), )
      score_records = sorted(score_records, key=lambda d: d[1], reverse=True)
      for k, v in score_records:
        print(k, v)

      print('-' * 80)
      print('Mean of Max scores:')
      print('-' * 80)
      score_records = []
      for transform in self.transforms:
        scores = transform.max_scores
        score = 0
        if scores:
          score = np.mean(scores)
        score_records.append((transform, score), )
      score_records = sorted(score_records, key=lambda d: d[1], reverse=True)
      for k, v in score_records:
        print(k, v)

      print('-' * 80)
      print('Max of Max scores:')
      print('-' * 80)
      score_records = []
      for transform in self.transforms:
        scores = transform.max_scores
        score = 0
        if scores:
          score = np.max(scores)
        score_records.append((transform, score), )
      score_records = sorted(score_records, key=lambda d: d[1], reverse=True)
      for k, v in score_records:
        print(k, v)

    # print('-' * 80)
    # for token, freq in sorted(self.kw_freq_dict.items(), key=lambda d: d[1], reverse=True)[:50]:
    #   print(token, freq)
    # print('Len freq dict:', len(self.kw_freq_dict))
    # print('-' * 80)
    return transformed_texts, local_scores


def generate_taa_samples(self, raw_texts, group_ids, rounds=5, topK=5):
  transformed_texts = []
  new_group_ids = []
  for raw_text, group_id in tqdm(zip(raw_texts, group_ids), total=len(raw_texts)):
    if isinstance(group_id, int):
      is_obs = (group_id == 1)
    else:
      is_obs = group_id.startswith('obs')

    texts_to_add = set()
    raw_tokens = self.tokenizer(raw_text)

    preprocessed_raw_text = preprocess_text(''.join(raw_tokens))
    historical_taa_set = {preprocessed_raw_text}
    candidate_taas = {preprocessed_raw_text: raw_tokens}
    for round in range(rounds):
      cur_tokens_list = [candidate_taas[text] for text in candidate_taas]
      for tokens_idx, tokens in enumerate(cur_tokens_list):
        if len(tokens) == 0:
          continue
        ## 遗传攻击
        for other_tokens_idx, other_tokens in enumerate(cur_tokens_list):
          if other_tokens_idx == tokens_idx or len(tokens) != len(other_tokens):
            continue

          new_tokens = tokens[:]
          target_token_indices = np.random.choice(len(other_tokens), len(other_tokens) // 2, replace=False)
          for idx in target_token_indices:
            if idx < len(new_tokens):
              new_tokens[idx] = other_tokens[idx]
          self._append_transformed_tokens(historical_taa_set, candidate_taas, new_tokens)

        idx = random.randint(0, len(tokens) - 1)  # Fixme: 换掉随机攻击

        if is_alpha(tokens[idx]) and len(tokens[idx]) >= 4:
          self._append_transformed_tokens(historical_taa_set, candidate_taas, self.char_swap_transform(tokens, idx))

        self._append_transformed_tokens(historical_taa_set, candidate_taas, self.add_transform(tokens, idx))
        self._append_transformed_tokens(historical_taa_set, candidate_taas, self.token_drop_transform(tokens, idx))
        self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                        self.token_swap_transform(tokens, idx))  # word lvl的swap很垃圾

        self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                        self.radical_transform(tokens, idx))  # 需要注意一些非左右结构的字，比如死、司等

        self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                        self.phonetic_char_swap_transform(tokens, idx))

        self._append_transformed_tokens(historical_taa_set, candidate_taas, self.hxw_transform(tokens, idx))

        # ## fixme: 下面这个是workflow中的小环节，属于特例
        candidates_list = self.pronunciation_transform(tokens, idx, N=None)
        transformed_tokens = tokens[:idx]
        new_token_chars = []
        for raw_char, candidates in zip(tokens[idx], candidates_list):
          for candidate in candidates:
            if candidate != raw_char:
              new_token_chars.append(candidate)
              break
        if len(new_token_chars) > 0:
          new_token = ''.join(new_token_chars)
        else:
          new_token = ''
        transformed_tokens.append(new_token)
        transformed_tokens += tokens[idx + 1:]
        self._append_transformed_tokens(historical_taa_set, candidate_taas, transformed_tokens)

      # 挑选出K个攻击力最强的样本，进行下一轮迭代
      cur_transformed_texts = []
      cur_transformed_tokens = []
      for text in candidate_taas:
        cur_transformed_texts.append(text)
        cur_transformed_tokens.append(candidate_taas[text])
      ref_texts = [raw_text] * len(cur_transformed_texts)
      soft_scores, hasrd_scores = self.performance_evaluator.calc_final_score(ref_texts, cur_transformed_texts,
                                                                              show_details=False, is_obs=is_obs)
      sorted_eval_scores = sorted(enumerate(soft_scores), key=lambda d: d[1], reverse=True)[:topK]
      candidate_taas = {}
      for idx, score in sorted_eval_scores:
        candidate_taas[cur_transformed_texts[idx]] = cur_transformed_tokens[idx]
      texts_to_add.add(cur_transformed_texts[sorted_eval_scores[0][0]])  # 每轮加一个最高分，最后一轮全加上
    texts_to_add |= set(cur_transformed_texts)
    transformed_texts.extend(list(texts_to_add))
    new_group_ids.extend([group_id] * len(texts_to_add))

  return transformed_texts, new_group_ids


def rule_based_transform(tokens, transform_dict):
  ## 筛选idx进行替换
  indices_probs = [transform_dict[token]['scores'] if token in transform_dict else 0.0
                   for token in tokens]
  indices_probs_sum = sum(indices_probs)
  if indices_probs_sum == 0:
    return []
  indices_probs = [prob / indices_probs_sum for prob in indices_probs]
  idx = np.random.choice(len(tokens), 1, p=indices_probs)[0]

  ## 对target_token进行替换
  new_tokens = tokens[:]
  target_token = new_tokens[idx]
  tsf_tokens = transform_dict[target_token]['transform_tokens']
  tsf_token_probs = transform_dict[target_token]['transform_probs']
  tsf_idx = np.random.choice(len(tsf_token_probs), 1, p=tsf_token_probs)[0]
  new_tokens[idx] = tsf_tokens[tsf_idx]
  return new_tokens


class RuleBasedAttacker():
  def __init__(self, transform_dict, attack_model, tokenizer):
    self.tokenizer = tokenizer
    self.transform_dict = transform_dict

    self.token_swap_transform = TokenSwapTransform()
    self.char_swap_transform = CharSwapTransform()
    self.add_transform = AddTransform()
    self.token_drop_transform = TokenDropTransform()
    self.char_drop_transform = CharDropTransform()

    self.phonetic_transform = PhoneticTransform()
    # self.phonetic_firstletter_transform = PhoneticTransform(first_letter=True)
    self.radical_transform = RadicalTransform('data/chaizi/chaizi-jt.txt')
    self.pronunciation_transform = PronunciationTransform('data/chaizi/中国所有汉字大全 一行一个.txt')
    self.homonymic_transform = HomonymicTransform()

    self.hxw_transform = HuoXingWenTransform()

    self.phonetic_char_swap_transform = SequentialModel([self.phonetic_transform, self.char_swap_transform])
    self.hxw_radical_transform = SequentialModel([self.hxw_transform, self.radical_transform])
    self.radical_chardrop_transform = SequentialModel([self.radical_transform, self.char_drop_transform])
    self.hxw_radical_chardroptransform = SequentialModel(
      [self.hxw_transform, self.radical_transform, self.char_drop_transform])

    vec_emb_path = 'data/materials/zh.300.vec.gz'
    self.performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=attack_model)  # 模拟远程防御模型，找到强力攻击样本

  def _append_transformed_tokens(self, historical_taa_set, candidate_taas, transformed_tokens):
    if not transformed_tokens:
      return
    preprocessed_transform_text = preprocess_text(''.join(transformed_tokens))
    if preprocessed_transform_text in historical_taa_set:
      return
    if preprocessed_transform_text not in candidate_taas:
      candidate_taas[preprocessed_transform_text] = transformed_tokens
      historical_taa_set.add(preprocessed_transform_text)

  def attack(self, raw_texts, rounds=5, topK=5):
    print('Round:', rounds, 'TopK:', topK)
    local_scores = []
    transformed_texts = []
    for raw_text in tqdm(raw_texts):
      best_score = 0.0
      raw_tokens = self.tokenizer(raw_text)
      best_transformed_text = raw_text
      best_transformed_tokens = raw_tokens

      preprocessed_raw_text = preprocess_text(''.join(raw_tokens))
      historical_taa_set = {preprocessed_raw_text}
      candidate_taas = {preprocessed_raw_text: raw_tokens}

      ##############################################################
      ### Global transform: 整个句子全都替换掉, 然后用这些样本当做种子
      ##############################################################
      ## 1. 暴力整句替换
      for _ in range(3):  # 调 3或5没什么区别，速度差一点点
        self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                        self.homonymic_transform.global_transform(raw_tokens))  # 替换掉所有骂人的关键词

      ## 2. 随机整句替换
      indices_probs = [self.transform_dict[token]['scores'] if token in self.transform_dict else 0.0
                       for token in raw_tokens]
      indices_probs_sum = 0
      valid_cnt = 0
      for prob in indices_probs:
        indices_probs_sum += prob
        valid_cnt += int(prob > 0)
      if indices_probs_sum > 0:
        indices_probs = [prob / indices_probs_sum for prob in indices_probs]
        for round in range(1):  # 这个轮数增多没有实际帮助
          for i in range(1, valid_cnt + 1):
            indices = np.random.choice(len(raw_tokens), i, replace=False, p=indices_probs)
            new_tokens = raw_tokens[:]
            for idx in indices:
              target_token = new_tokens[idx]
              tsf_tokens = self.transform_dict[target_token]['transform_tokens']
              tsf_token_probs = self.transform_dict[target_token]['transform_probs']
              tsf_idx = np.random.choice(len(tsf_token_probs), 1, p=tsf_token_probs)[0]
              new_tokens[idx] = tsf_tokens[tsf_idx]
            self._append_transformed_tokens(historical_taa_set, candidate_taas, new_tokens)

        # # 挑选出K个攻击力最强的样本，进行下一轮迭代
        # cur_transformed_texts = []
        # cur_transformed_tokens = []
        # for text in candidate_taas:
        #   cur_transformed_texts.append(text)
        #   cur_transformed_tokens.append(candidate_taas[text])
        # ref_texts = [raw_text] * len(cur_transformed_texts)
        # soft_scores, hard_scores = self.performance_evaluator.calc_final_score(ref_texts, cur_transformed_texts,
        #                                                                        show_details=False)
        # sorted_eval_scores = sorted(enumerate(soft_scores), key=lambda d: d[1], reverse=True)[:topK]
        # if sorted_eval_scores[0][1] > best_score:
        #   best_score = sorted_eval_scores[0][1]
        #   best_transformed_text = cur_transformed_texts[sorted_eval_scores[0][0]]
        #   best_transformed_tokens = cur_transformed_tokens[sorted_eval_scores[0][0]]
        #   candidate_taas = {}
        # else:
        #   candidate_taas = {best_transformed_text: best_transformed_tokens}
        # for idx, score in sorted_eval_scores:
        #   candidate_taas[cur_transformed_texts[idx]] = cur_transformed_tokens[idx]

      for round in range(rounds):
        cur_tokens_list = [candidate_taas[text] for text in candidate_taas]
        for tokens_idx, tokens in enumerate(cur_tokens_list):
          if len(tokens) == 0:
            continue
          ## 遗传攻击
          for other_tokens_idx, other_tokens in enumerate(cur_tokens_list):
            if other_tokens_idx == tokens_idx or len(tokens) != len(other_tokens):
              continue

            new_tokens = tokens[:]
            target_token_indices = np.random.choice(len(other_tokens), len(other_tokens) // 2, replace=False)
            for idx in target_token_indices:
              if idx < len(new_tokens):
                new_tokens[idx] = other_tokens[idx]
            self._append_transformed_tokens(historical_taa_set, candidate_taas, new_tokens)
          pass
          idx = random.randint(0, len(tokens) - 1)  # Fixme: 换掉随机攻击

          if is_alpha(tokens[idx]) and len(tokens[idx]) >= 4:
            self._append_transformed_tokens(historical_taa_set, candidate_taas, self.char_swap_transform(tokens, idx))

          self._append_transformed_tokens(historical_taa_set, candidate_taas, self.add_transform(tokens, idx))
          # self._append_transformed_tokens(historical_taa_set, candidate_taas, self.token_drop_transform(tokens, idx))
          self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                          self.radical_transform(tokens, idx))  # 需要注意一些非左右结构的字，比如死、司等

          self._append_transformed_tokens(historical_taa_set, candidate_taas, self.hxw_transform(tokens, idx))
          self._append_transformed_tokens(historical_taa_set, candidate_taas, self.hxw_radical_transform(tokens, idx))
          self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                          self.radical_chardrop_transform(tokens, idx))
          self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                          self.hxw_radical_chardroptransform(tokens, idx))

          # self._append_transformed_tokens(historical_taa_set, candidate_taas,
          #                                 self.token_swap_transform(tokens, idx))  # word lvl的swap很垃圾
          self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                          self.phonetic_char_swap_transform(tokens, idx))

          # # ## fixme: 下面这个是workflow中的小环节，属于特例
          # candidates_list = self.pronunciation_transform(tokens, idx, N=5)
          # transformed_tokens = tokens[:idx]
          # new_token_chars = []
          # for raw_char, candidates in zip(tokens[idx], candidates_list):
          #   for candidate in candidates:
          #     if candidate != raw_char:
          #       new_token_chars.append(candidate)
          #       break
          # if len(new_token_chars) > 0:
          #   new_token = ''.join(new_token_chars)
          # else:
          #   new_token = ''
          # transformed_tokens.append(new_token)
          # transformed_tokens += tokens[idx + 1:]
          # self._append_transformed_tokens(historical_taa_set, candidate_taas, transformed_tokens)
          self._append_transformed_tokens(historical_taa_set, candidate_taas,
                                          rule_based_transform(tokens, self.transform_dict))

        # 挑选出K个攻击力最强的样本，进行下一轮迭代
        cur_transformed_texts = []
        cur_transformed_tokens = []
        for text in candidate_taas:
          cur_transformed_texts.append(text)
          cur_transformed_tokens.append(candidate_taas[text])
        ref_texts = [raw_text] * len(cur_transformed_texts)
        soft_scores, hard_scores = self.performance_evaluator.calc_final_score(ref_texts, cur_transformed_texts,
                                                                               show_details=False)
        sorted_eval_scores = sorted(enumerate(soft_scores), key=lambda d: d[1], reverse=True)[:topK]
        if sorted_eval_scores[0][1] > best_score:
          best_score = sorted_eval_scores[0][1]
          best_transformed_text = cur_transformed_texts[sorted_eval_scores[0][0]]
          best_transformed_tokens = cur_transformed_tokens[sorted_eval_scores[0][0]]
          candidate_taas = {}
        else:
          candidate_taas = {best_transformed_text: best_transformed_tokens}
        for idx, score in sorted_eval_scores:
          candidate_taas[cur_transformed_texts[idx]] = cur_transformed_tokens[idx]

      transformed_texts.append(best_transformed_text)
      local_scores.append(best_score)

    return transformed_texts, local_scores


if __name__ == '__main__':
  bert_model_folder = 'ckpt/clf/ernie_weibo'
  bert_model = get_bert_inference_model(bert_model_folder, 32, 128)

  attack_model = bert_model
  defence_model = bert_model
  tokenizer = lambda x: list(x)

  obs_attacker = ObscenityAttacker(attack_model, defence_model, tokenizer)
  print(obs_attacker.attack(['你阿妈死了', 'nmsl']))
