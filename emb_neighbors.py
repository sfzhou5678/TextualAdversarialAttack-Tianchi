from gensim.models import KeyedVectors
from utils import pickle_dump
import json
from annoy import AnnoyIndex
import numpy as np

if __name__ == '__main__':
  EMBEDDING_PATH = 'data/materials/zh.300.vec.gz'
  DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=50000)

  id2word = {i: word for i, word in enumerate(DEFAULT_KEYVEC.index2word)}
  word2id = {word: i for i, word in enumerate(DEFAULT_KEYVEC.index2word)}

  n_trees = 100
  emb_dim = 300
  ann_index = AnnoyIndex(emb_dim, metric='angular')
  for i, word in enumerate(DEFAULT_KEYVEC.index2word):
    vec = DEFAULT_KEYVEC.get_vector(word)
    ann_index.add_item(i, vec)

  ann_index.build(n_trees)
  ann_index.save('data/index/annoy.cosine.idx')
  pickle_dump(id2word, 'data/index/id2word.pkl')
  pickle_dump(word2id, 'data/index/word2id.pkl')

  with open('data/index/annoy.cosine.10neighbors.txt', 'w', encoding='utf-8') as wf:
    for i, word in enumerate(DEFAULT_KEYVEC.index2word):
      cur_word = id2word[i]
      neighbors = [id2word[id] for id in ann_index.get_nns_by_item(i, 11)][1:]  # 第一个是自己，去掉
      wf.write('%s\t%s\n' % (cur_word, json.dumps(neighbors, ensure_ascii=False)))
