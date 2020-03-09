from .measure import normalized_levenshtein, jaccard_word, jaccard_char
from gensim.models import KeyedVectors
import numpy as np


def tokenize(text):
  import jieba
  return ' '.join(jieba.cut(text))


class DistanceCalculator:
  '''
  Computes pair-wise distances between texts, using multiple metrics.
  '''

  def __init__(self, EMBEDDING_PATH='distance_module/zh.300.vec.gz', EMBEDDING_DIM=300):
    self.EMBEDDING_DIM = EMBEDDING_DIM
    self.DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=50000)

  def __call__(self, docs_a, docs_b):
    docs_a_cut = [tokenize(_doc) for _doc in docs_a]
    docs_b_cut = [tokenize(_doc) for _doc in docs_b]

    # further validating input
    if not self.validate_input(docs_a, docs_b):
      raise ValueError("distance module got invalid input")

    # actual processing
    num_elements = len(docs_a)
    distances = dict()
    distances['normalized_levenshtein'] = [normalized_levenshtein(docs_a[i], docs_b[i]) for i in range(num_elements)]
    distances['jaccard_word'] = [jaccard_word(docs_a_cut[i], docs_b_cut[i]) for i in range(num_elements)]
    distances['jaccard_char'] = [jaccard_char(docs_a[i], docs_b[i]) for i in range(num_elements)]
    distances['embedding_cosine'] = self.batch_embedding_cosine_distance(docs_a_cut, docs_b_cut)
    return distances

  def doc2vec(self, tokenized):
    tokens = tokenized.split(' ')
    vec = np.full(self.EMBEDDING_DIM, 1e-10)
    weight = 1e-8
    for _token in tokens:
      try:
        vec += self.DEFAULT_KEYVEC.get_vector(_token)
        weight += 1.0
      except:
        pass
    return vec / weight

  def batch_doc2vec(self, list_of_tokenized_text):
    return [self.doc2vec(_text) for _text in list_of_tokenized_text]

  def validate_input(self, text_list_a, text_list_b):
    '''
    Determine whether two arguments are lists containing the same number of strings.
    '''
    if not (isinstance(text_list_a, list) and isinstance(text_list_b, list)):
      return False

    if not len(text_list_a) == len(text_list_b):
      return False

    for i in range(len(text_list_a)):
      if not (isinstance(text_list_a[i], str) and isinstance(text_list_b[i], str)):
        return False

    return True

  def batch_embedding_cosine_distance(self, text_list_a, text_list_b):
    '''
    Compute embedding cosine distances in batches.
    '''
    import numpy as np
    embedding_array_a = np.array(self.batch_doc2vec(text_list_a))
    embedding_array_b = np.array(self.batch_doc2vec(text_list_b))
    norm_a = np.linalg.norm(embedding_array_a, axis=1)
    norm_b = np.linalg.norm(embedding_array_b, axis=1)
    cosine_numer = np.multiply(embedding_array_a, embedding_array_b).sum(axis=1)
    cosine_denom = np.multiply(norm_a, norm_b)
    cosine_dist = np.divide(cosine_numer, cosine_denom)
    return cosine_dist.tolist()
