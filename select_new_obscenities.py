thres = 0.8

obscenities_set = set()
with open('data/obscenities.txt', encoding='utf-8') as f:
  for line in f:
    content = line.strip()
    if content not in obscenities_set:
      obscenities_set.add(content)
white_list_path = 'data/obscenities_white_list.txt'
white_list_set = set()
with open(white_list_path, encoding='utf-8') as f:
  for line in f:
    white_list_set.add(line.strip())

obscenities = []
with open('data/outputs/output.txt', encoding='utf-8') as f:
  for line in f:
    content, score = line.split('\t')
    if content in obscenities_set or content in white_list_set:
      continue
    score = float(score)
    if score >= thres:
      obscenities.append(content)
obscenities = list(set(obscenities))
with open('data/outputs/new_obscenities.txt', 'w', encoding='utf-8') as wf:
  for content in obscenities:
    wf.write(content + '\n')

print(obscenities)
print(len(obscenities))
