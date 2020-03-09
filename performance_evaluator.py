import numpy as np
from materials.distance_module import DistanceCalculator
from materials.preprocessing_module import preprocess_text


class PerformanceEvaluator():
  def __init__(self, vec_emb_path='data/materials/zh.300.vec.gz', defence_model=None):
    self.dc = DistanceCalculator(EMBEDDING_PATH=vec_emb_path)
    self.defence_model = defence_model

  def calc_final_score(self, ref_text_batch, transformed_text_batch, show_details=True, is_obs=True):
    assert len(ref_text_batch) == len(transformed_text_batch)
    ref_text_batch, transformed_text_batch = self._preprocess(ref_text_batch), self._preprocess(transformed_text_batch)

    if show_details:
      sim_batch, details = self._calc_sim_score(ref_text_batch, transformed_text_batch, show_details)
      probs = self._calc_prob(transformed_text_batch)
      for detail, prob in zip(details, probs):
        detail['prob'] = prob
        detail['final_score'] = (1 - prob) * detail['sim']
      if is_obs:
        soft_scores = sim_batch * (1 - probs)  # 对于obs样本, 我们希望模型输出的probs要尽量低(尽量分成0)
        hard_scores = sim_batch * (probs < 0.5)
      else:
        soft_scores = sim_batch * (probs)
        hard_scores = sim_batch * (probs > 0.5)
      return soft_scores, hard_scores, details
    else:
      sim_batch = self._calc_sim_score(ref_text_batch, transformed_text_batch, show_details)
      probs = self._calc_prob(transformed_text_batch)

      if is_obs:
        soft_scores = sim_batch * (1 - probs)
        hard_scores = sim_batch * (probs < 0.5)
      else:
        soft_scores = sim_batch * (probs)
        hard_scores = sim_batch * (probs > 0.5)
      return soft_scores, hard_scores

  def _calc_sim_score(self, ref_text_batch, transformed_text_batch, show_details):
    metrics = self.dc(ref_text_batch, transformed_text_batch)
    if show_details:
      details = []
      sim_batch = []
      for normalized_levenshtein, jaccard_word, jaccard_char, embedding_cosine in zip(metrics['normalized_levenshtein'],
                                                                                      metrics['jaccard_word'],
                                                                                      metrics['jaccard_char'],
                                                                                      metrics['embedding_cosine']):
        detail = {
          'normalized_levenshtein': normalized_levenshtein,
          'jaccard_word': jaccard_word,
          'jaccard_char': jaccard_char,
          'embedding_cosine': embedding_cosine,
          'sim': 3 / 14 * normalized_levenshtein + 1 / 7 * jaccard_word + 3 / 14 * jaccard_char + 3 / 7 * embedding_cosine
        }
        details.append(detail)
        sim_batch.append(detail['sim'])
      sim_batch = np.array(sim_batch)
      return sim_batch, details
    else:
      sim_batch = [3 / 14 * metrics['normalized_levenshtein'][i] + 1 / 7 * metrics['jaccard_word'][i] + \
                   3 / 14 * metrics['jaccard_char'][i] + 3 / 7 * metrics['embedding_cosine'][i]
                   for i in range(len(ref_text_batch))]
      sim_batch = np.array(sim_batch)
      return sim_batch

  def _calc_prob(self, transformed_text_batch):
    obscene_probs = self.defence_model.inference(transformed_text_batch)
    return obscene_probs

  def _preprocess(self, texts):
    texts = [preprocess_text(text) for text in texts]
    return texts


if __name__ == '__main__':
  from model import FastTextInferenceModel, get_bert_inference_model, get_fasttext_inference_model
  from attackers import ObscenityAttacker
  from pytorch_pretrained_bert import BertTokenizer

  max_line = 200
  # inp_path = 'data/obscenities.txt'
  # inp_lines = set()
  # with open(inp_path, 'r', encoding='utf-8') as f:
  #   for line in f:
  #     inp_lines.add(line.strip())
  #     if len(inp_lines) >= max_lines:
  #       break
  # inp_lines = list(inp_lines)

  import random
  from sklearn.model_selection import GroupShuffleSplit

  train_file = 'data/clf/train_adv0.txt'
  seed = 12345
  train_samples = []
  content_set = set()
  with open(train_file, encoding='utf-8') as f:
    for line in f:
      sample = line.strip().split('\t')
      if len(sample) >= 3 and sample[0] not in content_set:
        train_samples.append(sample)
        content_set.add(sample[0])
      elif len(sample) >= 2 and not (sample[-1].startswith('obs') or sample[-1].startswith('non')) and sample[
        0] not in content_set:
        sample.append('random_%d' % len(train_samples))
        train_samples.append(sample)
        content_set.add(sample[0])
  random.seed = seed
  random.shuffle(train_samples)
  group_ids = [sample[-1] for sample in train_samples]
  train_samples = [sample[:2] for sample in train_samples]
  folds = GroupShuffleSplit(n_splits=5, random_state=seed)
  for fold, (trn_idx, val_idx) in enumerate(folds.split(train_samples, groups=group_ids)):
    val_group_ids = [group_ids[idx] for idx in val_idx]
    val_id_set = set(val_group_ids)
    break

  inp_lines = []
  obscenities = []
  content_set = set()
  with open('data/obscenities.txt', encoding='utf-8') as f:
    for line in f:
      content = line.strip()
      if content not in content_set:
        obscenities.append((content, 1, 'obs_%d' % len(obscenities)))
        if obscenities[-1][-1] in val_id_set:
          inp_lines.append(obscenities[-1][0])
        content_set.add(content)
      if max_line is not None and len(obscenities) >= max_line:
        break

  rounds = 20
  topK = 10

  ########################################################################################
  ## 先使用demo model做防御，目标是完全攻破这个模型
  ########################################################################################
  fasttext_model_path = 'data/materials/mini.ftz'
  fasttext_model = FastTextInferenceModel(fasttext_model_path)
  remote_defence_model = fasttext_model

  # local_fastttext_model_path = 'ckpt/clf/fasttext_simple_iter_adv0'
  # local_fasttext_model = get_fasttext_inference_model(local_fastttext_model_path, 128, 100)
  # remote_defence_model = local_fasttext_model

  # fasttext_model_path = 'ckpt/pip_fasttext/bert_tokenizer_based.ftz'
  # fasttext_model = FastTextInferenceModel(fasttext_model_path)
  # remote_defence_model = fasttext_model

  # ------------
  ########################################################################################
  ### 攻击模型
  ########################################################################################
  # fasttext_model_path = 'data/materials/mini.ftz'
  # fasttext_model = FastTextInferenceModel(fasttext_model_path)
  # kw_identify_model = fasttext_model
  # attack_model = fasttext_model

  local_fastttext_model_path = 'ckpt/clf/fasttext_simple_mixup_adv0.0'
  local_fasttext_model = get_fasttext_inference_model(local_fastttext_model_path, 128, 100)
  kw_identify_model = local_fasttext_model
  attack_model = local_fasttext_model

  # attack_bert_model_path = 'ckpt/clf/ernie_adv0'
  # attack_model = get_bert_inference_model(attack_bert_model_path, 128, 100)  # 本地对抗训练得到的防御模型
  # kw_identify_model = attack_model

  # attack_bert_model_path = 'ckpt/clf/ernie_adv2'
  # attack_model = get_bert_inference_model(attack_bert_model_path, 128, 100)  # 本地对抗训练得到的防御模型

  # remote_bert_model_path = 'ckpt/clf/ernie_weibo'
  # remote_defence_model = get_bert_inference_model(remote_bert_model_path, 128, 100)  # 模拟远程的模型，可以用某个早期的防御模型，用来测试对抗模型的效果

  bert_tokenizer = BertTokenizer.from_pretrained('data/chinese_vocab.txt', do_lower_case=True)
  tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)
  ########################################################################################

  # %%
  obs_attacker = ObscenityAttacker(kw_identify_model, attack_model, tokenizer)
  out_lines, local_scores = obs_attacker.attack(inp_lines, rounds, topK)
  print('Local model avg score:', sum(local_scores) / len(local_scores))

  vec_emb_path = 'data/materials/zh.300.vec.gz'
  performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=remote_defence_model)
  soft_scores, hard_scores = performance_evaluator.calc_final_score(inp_lines, out_lines, show_details=False)
  print('Remote model soft score:', sum(soft_scores) / len(soft_scores))
  print('Remote model hard score:', sum(hard_scores) / len(hard_scores))

  # for remote_defence_model in [
  #
  # ]:
  #   remote_scores = performance_evaluator.calc_final_score(inp_lines, out_lines, show_details=False)
  #   print('Remote model avg score:', sum(remote_scores) / len(remote_scores))
  #
  # remote_bert_model_path = 'ckpt/clf/ernie_weibo'
  # remote_defence_model = get_bert_inference_model(remote_bert_model_path, 128, 100)  # 模拟远程的模型，可以用某个早期的防御模型，用来测试对抗模型的效果
  # performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=remote_defence_model)
  # remote_scores = performance_evaluator.calc_final_score(inp_lines, out_lines, show_details=False)
  # print('Weibo Remote model avg score:', sum(remote_scores) / len(remote_scores))
  #
  # remote_bert_model_path = 'ckpt/clf/ernie_adv1'
  # remote_defence_model = get_bert_inference_model(remote_bert_model_path, 128, 100)  # 模拟远程的模型，可以用某个早期的防御模型，用来测试对抗模型的效果
  # performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=remote_defence_model)
  # remote_scores = performance_evaluator.calc_final_score(inp_lines, out_lines, show_details=False)
  # print('adv1 Remote model avg score:', sum(remote_scores) / len(remote_scores))
