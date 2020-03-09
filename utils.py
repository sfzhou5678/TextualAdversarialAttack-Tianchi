import pickle
import re


def pickle_dump(data, file_path):
  f_write = open(file_path, 'wb')
  pickle.dump(data, f_write, True)


def pickle_load(file_path):
  f_read = open(file_path, 'rb')
  data = pickle.load(f_read)

  return data


def is_alpha(s, alpha_pattern='^[a-zA-Z\\s012⚬θαενγ_|`~-]+$'):  # pattern中加入了自己新增的特殊替换符号以及sep符号
  return re.match(alpha_pattern, s) is not None


if __name__ == '__main__':
  pass
