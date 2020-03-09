thres = 0.95

obscenities_set = set()
white_list_set = set()

obscenities = []
with open('corpus_output.txt', encoding='utf-8') as f:
  for line in f:
    content, score = line.split('\t')
    if content in obscenities_set or content in white_list_set:
      continue
    score = float(score)
    if score >= thres:
      obscenities.append(content)

obscenities = list(set(obscenities))

with open('new_obscenities.txt', 'w', encoding='utf-8') as wf:
  for content in obscenities:
    wf.write(content + '\n')

# print(obscenities)
print(len(obscenities))
