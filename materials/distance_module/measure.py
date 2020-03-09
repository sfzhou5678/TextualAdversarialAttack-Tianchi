'''
Distance measures between strings.
'''
import editdistance


def normalized_levenshtein(str_a, str_b):
  '''
  Edit distance normalized to [0, 1].
  '''
  return 1 - min(editdistance.eval(str_a, str_b) / (len(str_a) + 1e-16), 1.0)


def jaccard_set(set_a, set_b):
  '''
  Jaccard SIMILARITY between sets.
  '''
  set_c = set_a.intersection(set_b)
  return float(len(set_c)) / (len(set_a) + len(set_b) - len(set_c) + 1e-16)


def jaccard_char(str_a, str_b):
  '''
  Jaccard DISTANCE between strings, evaluated by characters.
  '''
  set_a = set(str_a)
  set_b = set(str_b)
  return jaccard_set(set_a, set_b)


def jaccard_word(str_a, str_b, sep=' '):
  '''
  Jaccard DISTANCE between strings, evaluated by words.
  '''
  set_a = set(str_a.split(sep))
  set_b = set(str_b.split(sep))
  return jaccard_set(set_a, set_b)
