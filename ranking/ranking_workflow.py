@DeprecationWarning
class TaaSampleRanker():
  """
  TaaSampleRanker评估给定的tokens、要变形的idx和所有变形方法的candidates的对抗样本质量：
  1. 用LM ranker计算在idx处应填入的词，用于排序candidates
  2. 将各对抗样本代入本地的defence model来评估TAA samples的质量
  3. 最后结合上述得分完成最终排序，可以选得分最高的作为最终的对抗样本。
  """

  def __init__(self, lm_ranker, defence_model):
    self.lm_ranker = lm_ranker
    self.defence_model = defence_model

    # self.eval_func = lambda lm_score, obscene_prob: (lm_score + (1 - obscene_prob)) / 2
    self.eval_func = lambda lm_score, obscene_prob: max(lm_score, 1 - obscene_prob)

  def rank(self, tokens, idx, candidates):
    ## fixme: 现在的无法处理多个词的替换, 比如妈->女马或者ma这种。
    if self.lm_ranker is not None:
      relevance_dict = self.lm_ranker(tokens, idx)  # 记录了当前idx下各个char出现的概率
    else:
      relevance_dict = {}
    transformed_text_batch = []
    for candidate in candidates:
      transformed_tokens = tokens[:]
      transformed_tokens[idx] = candidate
      transformed_text_batch.append(''.join(transformed_tokens))
    obscene_probs = self.defence_model.inference(transformed_text_batch)

    final_scores = []
    for candidate, obscene_prob in zip(candidates, obscene_probs):
      lm_score = 0
      # if candidate in relevance_dict:
      #   lm_score = relevance_dict[candidate]
      final_scores.append(self.eval_func(lm_score, obscene_prob))

    return final_scores


if __name__ == '__main__':
  from model import get_bert_inference_model
  from ranking import get_lm_ranker

  # max_seq_length = 128
  # lm_bert_model = '../ckpt/raw/bert-base-chinese'
  # lm_ranker = get_lm_ranker(lm_bert_model, max_seq_length)
  lm_ranker = None

  bert_model = '../ckpt/clf/ernie_weibo'
  inference_model = get_bert_inference_model(bert_model)

  taa_sample_ranler = TaaSampleRanker(lm_ranker=lm_ranker, defence_model=inference_model)
  tokens = list('你妈死了')
  idx = 1
  candidates = ['马', '女马', 'ma', 'm']
  print(taa_sample_ranler.rank(tokens, idx, candidates))
