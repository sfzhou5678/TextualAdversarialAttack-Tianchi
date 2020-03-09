class SingleCharIdentification():
  def __init__(self, inference_model):
    self.inference_model = inference_model

  def __call__(self, raw_tokens, N=2):
    """
    通过drop掉第i个词汇达到识别关键词的效果
    :param raw_tokens:
    :param N:
    :return:
    """
    text_batch = []
    for i in range(len(raw_tokens)):
      new_tokens = raw_tokens[:i] + raw_tokens[i + 1:]
      text_batch.append(''.join(new_tokens))
    base_prob = self.inference_model.inference([''.join(raw_tokens)])[0]
    obsecne_probs = self.inference_model.inference(text_batch)
    probs_diff = [(i, base_prob - prob) for i, prob in enumerate(obsecne_probs)]
    probs_diff = sorted(probs_diff, key=lambda d: d[1], reverse=True)

    return probs_diff[:N]


if __name__ == '__main__':
  pass
