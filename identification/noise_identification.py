"""
根据token scores从低到高，选出不再emb vocab中的前k个tokens当做noise
"""
from utils import pickle_load

if __name__ == '__main__':
  token_scores_path = 'token_scores_dict.fasttext.bert_tokenizer.json'
  word_vocab = pickle_load('../data/index/word2id.pkl')
  token_scores = []
  with open(token_scores_path, encoding='utf-8') as f:
    for line in f:
      try:
        token, score = line.strip().split('\t')
        score = float(score)
        if token not in word_vocab:
          token_scores.append((token, score))
      except:
        print(line)

  for token, score in token_scores[-50:]:
    print(token, score)
