# encoding:utf-8
import os
import re
import jieba

pattern = '(/.*)$'
with open('data/weibo/12w/corpus_segmentation.txt', 'w', encoding='utf-8') as wf:
  for file in os.listdir('data/weibo/12w'):
    with open(os.path.join('data/weibo/12w', file), encoding='gb18030') as f:
      for line in f:
        for content in line.strip().split('//@'):
          idx = content.find(': ')
          if idx >= 0:
            content = content[idx + 2:].strip()
            words = jieba.cut(content)
            s = ' '.join(words)
            if s != '':
              wf.write(s + '\n')
