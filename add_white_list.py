import os
import shutil

white_list_path = 'data/obscenities_white_list.txt'
white_list_backup_path = 'data/obscenities_white_list.backup.txt'
shutil.copyfile(white_list_path, white_list_backup_path)

thres = 0.8

content_set = set()
with open('data/obscenities.txt', encoding='utf-8') as f:
  for line in f:
    content = line.strip()
    if content not in content_set:
      content_set.add(content)

new_white_set = set()
if os.path.exists('data/outputs/output.txt'):
  with open('data/outputs/output.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      content, score = line.split('\t')
      score = float(score)
      if score >= thres and content not in content_set:
        new_white_set.add(content)

white_set = set()
with open(white_list_path, encoding='utf-8') as f:
  for line in f:
    white_set.add(line.strip())

white_set |= new_white_set
with open('data/obscenities_white_list.txt', 'w', encoding='utf-8') as wf:
  for line in white_set:
    wf.write(line + '\n')
