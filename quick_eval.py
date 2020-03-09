import json
from model import FastTextInferenceModel, get_bert_inference_model
from pytorch_pretrained_bert import BertTokenizer
from performance_evaluator import PerformanceEvaluator
from attackers import RuleBasedAttacker

if __name__ == '__main__':
  max_lines = 300
  inp_path = 'data/obscenities.txt'
  out_path = 'adversarial.txt'

  inp_lines = []
  with open(inp_path, 'r', encoding='utf-8') as f:
    for line in f:
      inp_lines.append(line.strip())
      if len(inp_lines) >= max_lines:
        break
  out_lines = json.load(open(out_path, encoding='utf-8'))['text']

  if True:
    inp_lines = inp_lines[:min(len(inp_lines), len(out_lines))]
    out_lines = out_lines[:min(len(inp_lines), len(out_lines))]
    print('Lines:', len(out_lines))

  fasttext_model_path = 'data/materials/mini.ftz'
  # fasttext_model_path = ['data/materials/mini.ftz','data/materials/mini-explicit-labels.ftz']
  fasttext_model = FastTextInferenceModel(fasttext_model_path)
  remote_defence_model = fasttext_model
  # bert_model_folder = 'ckpt/clf/ernie_weibo'
  # remote_defence_model = get_bert_inference_model(bert_model_folder, 128, 100)

  # bert_tokenizer = BertTokenizer.from_pretrained('data/chinese_vocab.txt', do_lower_case=True)
  # tokenizer = lambda x: bert_tokenizer.basic_tokenizer.tokenize(x)

  vec_emb_path = 'data/materials/zh.300.vec.gz'
  performance_evaluator = PerformanceEvaluator(vec_emb_path, defence_model=remote_defence_model)
  soft_scores, hard_scores = performance_evaluator.calc_final_score(inp_lines, out_lines, show_details=False)
  print('Remote model soft score:', sum(soft_scores) / len(soft_scores))
  print('Remote model hard score:', sum(hard_scores) / len(hard_scores))
