class ProbCalculator():
  def __init__(self):
    ## fixme: 静态部分改成离线计算好的dict
    from annoy import AnnoyIndex
    from utils import pickle_load
    self.ann_index = AnnoyIndex(300, metric='euclidean')
    self.ann_index.load('data/index/annoy.euclidean.idx')
    self.id2word = pickle_load('data/index/id2word.pkl')
    self.word2id = pickle_load('data/index/word2id.pkl')

  def calc_probs(self, char, candidates):
    static_probs = self._static_probs(char, candidates)
    probs = static_probs  # 下面的动态prob没卵用
    # if static_probs is not None:
    #   dynamic_probs = self._dynamic_probs(candidates, tokens, idx)
    #   joint_probs = [0.4 * sp + 0.6 * dp for sp, dp in zip(static_probs, dynamic_probs)]
    #   probs_sum = sum(joint_probs)
    #   if probs_sum > 0:
    #     probs = [p / probs_sum for p in joint_probs]
    return probs

  def _static_probs(self, char, candidates):
    """
    计算静态的emb相似度
    :param char:
    :param candidates:
    :return:
    """
    probs = None
    if char in self.word2id:
      # char_vec = np.array(self.ann_index.get_item_vector(self.word2id[char]))
      # unk_sim = 1 / np.sqrt(sum(char_vec ** 2))  # unk相当于vec=np.zeros, 所以dis=|v[char]|, 但在欧式距离下，这个sim会比大部分非unk的词要高
      unk_sim = 0  #

      probs_sum = 0
      probs = []
      for candidate in candidates:
        sim = unk_sim
        if candidate in self.word2id:
          sim = 1 / self.ann_index.get_distance(self.word2id[char], self.word2id[candidate])  # 上面的代码已经过滤掉了dis=0的情况
        probs.append(sim)
        probs_sum += sim
      if probs_sum > 0:
        probs = [p / probs_sum for p in probs]
      else:
        probs = None
    return probs

  def _dynamic_probs(self, candidates, tokens, idx):
    """
    根据上下文计算candidate对edit distance和jaccard的影响
    :param char:
    :param candidates:
    :param tokens:
    :param idx:
    :return:
    """
    # target_token = tokens[idx]
    # t_set = set(target_token)
    # len_target_token = len(target_token)
    # sentence = ''.join(tokens)
    # probs = []
    # for candidate in candidates:
    #   if len(candidate) == 0:
    #     probs.append(0.5)
    #     continue
    #   c_set = set(candidate)
    #   delta_len = len(c_set) - len(c_set & t_set)
    #   edit_sim = 1 - delta_len / (len(sentence) + 1e-8)
    #   jaccard_sim = len(c_set & set(sentence)) / len(c_set)
    #   sim = 0.4 * edit_sim + 0.6 * jaccard_sim
    #   probs.append(sim)
    probs = []
    s_set = set(''.join(tokens))
    for candidate in candidates:
      prob = 0
      for c in candidate:
        if c in s_set:
          prob = 1
          break
      probs.append(prob)
    probs_sum = sum(probs)
    if probs_sum > 0:
      probs = [p / probs_sum for p in probs]
    else:
      probs = [1] * len(candidates)
    return probs


prob_calculator = ProbCalculator()
